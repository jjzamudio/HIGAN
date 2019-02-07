import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import h5py
import matplotlib as mpl
from pathlib import Path
import utils.data_utils
import timeit




def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Function for dividing/truncating cmaps"""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



def visualize_cube(cube=None,      # array name
             edge_dim=None,        # edge dimension (128 for 128 x 128 x 128 cube)
             start_cube_index_x=0,
             start_cube_index_y=0,
             start_cube_index_z=0,
             fig_size=None,
#              stdev_to_white=-3,
             norm_multiply=1e2,
             size_magnitude = False,
             save_fig = False,
            raw_cube_max = False,
                  inverse_transform = False):

    """Takes as input;
    - cube: A 3d numpy array to visualize,
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
#     cube = cube/raw_cube_max

    
    time_start = timeit.default_timer()
        
    cube_size = edge_dim
    edge = np.array([*range(cube_size)])
    
    end_x = start_cube_index_x + cube_size
    end_y = start_cube_index_y + cube_size
    end_z = start_cube_index_z + cube_size
    
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

    X, Y, Z, data_1dim = [axis[np.where(data_1dim>0)] for axis in [X,Y,Z,data_1dim]]

    if size_magnitude == True:
        s = norm_multiply * data_1dim
    else:
        s = norm_multiply * np.ones_like(a = data_1dim)

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
        #ax.xaxis.pane.fill = False
        #ax.yaxis.pane.fill = False
        #ax.zaxis.pane.fill = False
        
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
        
#         print(inverse_transform)
#         if inverse_transform == "log_scale_01":
#             _vmin = 0.0
#             _vmax = 0.25
#         elif inverse_transform == "log_scale_neg11":
#             _vmin = -1.0
#             _vmax = -0.5
#         else:
#             raise NotImplementedError

        _vmin = 0.0
        _vmax = 0.25
            
#         print("_vmin = " + str(_vmin))
#         print("_vmax = " + str(_vmax))
            
        ax.scatter(X, Y, Z,      
                   c=data_1dim, 
                   cmap=new_cmap,
                   s=2, # Juan = 2          
                   alpha = 1, # Juan = 1
                   vmin=_vmin,
                   vmax=_vmax,
                   edgecolors="face")
        
        plt.show()
        
        if save_fig != '':
            fig.savefig(save_fig, bbox_inches='tight', dpi = 250)

        plt.close(fig)
    

    
    
def mmd_hist_plot(noise, real, recon_noise, recon_real,
                  epoch, file_name, 
                  plot_pdf , log_plot, plot_show,
                 redshift_fig_folder):
    """
    Args:
        recon(): generated data
        real(): real data
        epoch(integer): epoch number
        file_name(string): name of the file
        hd (integer) : if 0 it's a histogram, if 1 it's a pdf
        
    """
    if log_plot:
        try:
            noise = np.log10(noise)
            real = np.log10(real)
            recon_noise = np.log10(recon_noise)
            recon_real = np.log10(recon_real)
        except:
            print("Couldnt take the log of the values...")
            return
    
    
    plt.figure(figsize = (16,8))
    if plot_pdf == False:
        plt.title("Histograms of Hydrogen")
    else:
        plt.title("PDFs of Hydrogen")
        
    plot_min = min(noise.min(),real.min(),recon_noise.min(),recon_real.min())
    plot_max = max(noise.max(),real.max(),recon_noise.max(),recon_real.max())
#     print("plot_min = " + str(plot_min))
#     print("plot_max = " + str(plot_max))
#     plt.xlim(plot_min,plot_max)
    plt.xlim(0.01, .4)
    plt.ylim(0, 20)
    
#     if not log_plot:
    bins = np.linspace(plot_min,plot_max,300)
#     if log_plot:
#         bins = np.logspace(np.log10(plot_min),np.log10(plot_min), 300)
    
    real_label = "Real Subcube - Only Nonzero"
    noise_label = "Noise Subcube - Only Nonzero"
    recon_noise_label = "Reconstructed Noise Subcube - Only Nonzero"
    recon_real_label = "Reconstructed Real Subcube - Only Nonzero"

    if log_plot:
        real_label = real_label + "  (Log10)"
        noise_label = noise_label + "  (Log10)"
        recon_noise_label = recon_noise_label + "  (Log10)"
        recon_real_label = recon_real_label + "  (Log10)"
                  
    plt.hist(real, 
             bins = bins, 
             color = "blue" ,
             log = log_plot,
             alpha = 0.5, 
             label = real_label,
             density = plot_pdf)
    plt.hist(recon_real, 
             bins = bins, 
         color = "lightblue" ,
             log = log_plot,
         alpha= 0.2, 
         label = recon_real_label,
            density = plot_pdf)
    plt.hist(noise, 
             bins = bins, 
         color = "red" ,
             log = log_plot,
         alpha= 0.5, 
         label = noise_label,
            density = plot_pdf)
    plt.hist(recon_noise, 
             bins = bins, 
         color = "orange" ,
             log = log_plot,
         alpha= 0.2, 
         label = recon_noise_label,
            density = plot_pdf)


    plt.legend()
    if log_plot:
        plt.savefig(redshift_fig_folder + file_name , 
                    bbox_inches='tight')
    else:
        plt.savefig(redshift_fig_folder + file_name, 
                bbox_inches='tight')
    
    if plot_show:
        plt.show() 
    plt.close()

    return



def mmd_loss_plots(fig_id, fig_title, data, show_plot, save_plot, redshift_fig_folder, t, dist_ae):
    """
    Args:
        fig_id(int): figure number
        fig_title(string): title of the figure
        data(): data to plot
        save_direct(string): directory to save
    """
    plt.figure(fig_id, figsize = (10,5))
    
    # adjusting the plot title of Reconstruction Errors based on L1 or L2
    if fig_title in ["L2_AE_X_D_list","L2_AE_Y_D_list"]:
#         if dist_ae not in ["L2_AE_X_D_list", "L2_AE_Y_D_list"]:
        if dist_ae != "L2":
            fig_title = fig_title.replace("L2","L1")
    plt.title(fig_title)
    
    # taking log10 of some dat for better viewing
#     if fig_title in ["mmd2_D_before_ReLU_list", 
#                      "mmd2_D_after_ReLU_list",
#                      "L2_AE_X_D_list",
#                      "L2_AE_Y_D_list",
#                      "mmd2_G_before_ReLU_list", 
#                      "mmd2_G_after_ReLU_list"]:
#         try:
#             data = np.log(np.array(data))
#             plt.ylabel("log("+fig_title+")")
#         except:
#             pass
#     else:
#         plt.ylabel(fig_title)
        
    plt.xlabel("Number of Minibatch Iterations")
    
    plt.plot(data,
             linewidth=0.5)
    if save_plot:
        plt.savefig(redshift_fig_folder + fig_title +'_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show()
        
    plt.close()

    
def plot_minibatch_value_sum(sum_real,       
                             sum_real_recon,
                             sum_noise_gen,
                             sum_noise_gen_recon,
                             save_plot,
                             show_plot,
                             redshift_fig_folder,
                             t):
    """
    All the input the data should be inverse transformed to make it comparable with
    other transformation types.
    """              
    plt.figure(figsize = (12,6))
    plt.title("Sum of Minibatches (inverse transformed) in Log10 Scale")
    plt.xlabel("Epochs")
    plt.plot(np.log10(sum_real), 
             label = "sum_real", 
             alpha = 0.9,
             color = "blue")
    plt.plot(np.log10(sum_real_recon), 
             label = "sum_real_recon", 
             alpha = 0.3,
             color = "lightblue")
    plt.plot(np.log10(sum_noise_gen), 
             label = "sum_noise_gen", 
             alpha = 0.9,
             color = "red")
    plt.plot(np.log10(sum_noise_gen_recon), 
             label = "sum_noise_gen_recon", 
             alpha = 0.3,
             color = "orange")
    plt.legend()
    if save_plot:
        plt.savefig(redshift_fig_folder + 'sum_minibatch_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show() 
    
    plt.close()
    
    
    
def visualize2d(real, fake, raw_cube_mean, scaling, t,
                redshift_fig_folder = False,
                save_plot = False, show_plot = False):
    """
    real = 5d tensor
    fake = 5d tensor
    
    np.nonzero is not used when taking the log -> changes shape of input
    eps = 1e-36 is added to plot
    """
    
#   min_,  max_ = 0, raw_cube_mean 
#     if scaling == "log_scale_01":
#         min_, max_ = 0.0 , 0.22 # for log + 0,1 scaling
#     elif scaling == "log_scale_neg11":
#         min_, max_ = -1.0 , -0.8 # for log + -1,1 scaling
#     else:
#         raise NotImplementedError
        
    min_, max_ = 0.0 , 0.22 # for log + 0,1 scaling
        
#     print("max_ = "  +str(max_))
#     cols = real.shape[0] // (2*4)
    cols = 8
    rows = 2
    
    fig, axes = plt.subplots(nrows=2, 
                             ncols=cols, 
                             figsize=(16,4))
    
    for ax, row in zip(axes[:,0], ['Generated', 'Real']):
        ax.set_ylabel(row, rotation=90, fontsize=16)
        
#     print("Fake 2D: " + str(fake[0][0].mean(axis=1)))
#     print("Real 2D: " + str(real[0][0].mean(axis=1)))

#     print("fake = " + str(fake))

    print(fake.shape)
    print(real.shape)
    
    m = 0
    for ax in axes.flat:
        #Plot only half of the mini-batch
        if m < cols:
#             print("fake max = "  +str(np.max(fake[m][0])))
#             print("fake min = "  +str(np.min(fake[m][0])))
#             print(fake.size())
#             print(m)

            if len(fake.shape) == 5:
                plot_cube = fake[m][0].mean(axis=1)
            else:
                plot_cube = fake[m][0]
            
#             if len(fake.shape) == 5:
#                 plot_cube = fake[m].mean(axis=1)
#             else:
#                 plot_cube = fake[m]
                
            plot_cube = plot_cube + 1.0
            im = ax.imshow(np.log(plot_cube), aspect='equal', 
                       interpolation=None, vmin=min_, vmax=max_)
        
        else:
#             print("real max = "  +str(np.max(real[m][0])))
#             print("real min = "  +str(np.min(real[m][0])))

#             print(m)
    
            if len(real.shape) == 5:
                plot_cube = real[m-cols][0].mean(axis=1)
            else:
                plot_cube = real[m-cols][0]
                
#             if len(real.shape) == 5:
#                 plot_cube = real[m-cols].mean(axis=1)
#             else:
#                 plot_cube = real[m-cols]
                
            plot_cube = plot_cube + 1.0
            im = ax.imshow(np.log(plot_cube), aspect='equal', 
                       interpolation=None, vmin=min_, vmax=max_)
        m += 1
        ax.set_xticks([])
        ax.set_yticks([])
    #fig.subplots_adjust(right=.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(ax, cax=cbar_ax)
    fig.tight_layout() 
    
    if save_plot:
        plt.savefig(redshift_fig_folder + 'proj_2d_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show() 
    
    plt.close()    
    
    
    
def hist_plot_multiple(noise, real, log_plot, redshift_fig_folder, t, save_plot, show_plot) :
    """
    Args:
        real(): real data
        epoch(integer): epoch number
        file_name(string): name of the file
        hd (integer) : if 0 it's a histogram, if 1 it's a pdf
        
    """
    batch_size = noise.shape[0]
    
#     plt.figure(figsize = (20,10))
    plt.figure(figsize = (12,6))
    plot_min = min(float(noise.min()), float(real.min()))
    plot_max = max(float(noise.max()), float(real.max()))
#     plt.xlim(0.01, 0.5)
    plt.xlim(plot_min, plot_max)
    plt.ylim(0, 20)
    font_size = 12
    
#     bins = np.linspace(0, 0.5, 500)
#     bins = np.linspace(plot_min, 0.5, 500)
    bins = np.linspace(plot_min, plot_max, 500)
    
    for m in range(min(batch_size,16)):
        
        if len(real.shape) == 5:
            real_viz = real[m][0].flatten()
            noise_viz = noise[m][0].flatten()
        elif len(real.shape) == 4:
            real_viz = real[m].flatten()
            noise_viz = noise[m].flatten()   
        else:
            raise NotImplementedError
        
        if m==0:
            plt.hist(real_viz, bins = bins, 
                     color = "b" , log = log_plot, alpha = 0.05, 
                     density=True, label='Real')
            plt.hist(noise_viz, bins = bins, 
                     color = "r" , log = log_plot, alpha= 0.05, 
                     density=True, label='Generated')
        else:
            plt.hist(real_viz, bins = bins, 
                     color = "b" , log = log_plot, alpha = 0.05, 
                     density=True)
            plt.hist(noise_viz, bins = bins, 
                     color = "r" , log = log_plot, alpha= 0.05, 
                     density=True)
            
  
    plt.tick_params(axis='both', labelsize=font_size)
    plt.title('Densities of Real and Generated Samples', fontsize=font_size)
    plt.legend(fontsize=font_size)
    
    if save_plot:
        plt.savefig(redshift_fig_folder + "hist_allbatch_" + str(t) + ".png", 
                bbox_inches='tight') 
    
    if show_plot:
        plt.show() 
        
    plt.close()
    
    
    
def histogram_mean_confint(noise, real, log_plot, redshift_fig_folder, t, save_plot, show_plot):
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
    plt.figure(figsize = (12,6))
    plot_min = min(float(noise.min()), float(real.min()))
    plot_max = max(float(noise.max()), float(real.max()))
#     plt.xlim(0.01, 0.5)
    plt.xlim(plot_min, plot_max)
    plt.ylim(0, 20)
    font_size = 12
    
#     bins = np.linspace(0, 0.5, 500)
#     bins = np.linspace(plot_min, 0.5, 500)
    bins = np.linspace(plot_min, plot_max, 500)
    
        
    for m in range(min(batch_size,16)):
#         print(m)

        if len(real.shape) == 5:
            real_viz = real[m][0].flatten()
            noise_viz = noise[m][0].flatten()
        elif len(real.shape) == 4:
            real_viz = real[m].flatten()
            noise_viz = noise[m].flatten()   
        else:
            raise NotImplementedError

        bin_vals_real, _ , _ = plt.hist(real_viz, bins = bins, 
                                 color = "b" , log = False, alpha = 0.00, 
                                 density=True, label='Real')
        bin_vals_noise, _ , _ = plt.hist(noise_viz, bins = bins, 
                             color = "b" , log = False, alpha = 0.00, 
                             density=True, label='Generated')

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
    
            
  
    plt.tick_params(axis='both', labelsize=font_size)
    plt.title('Means and Confidence Intervals of Real and Generated Samples', fontsize=font_size)
#     plt.legend(fontsize=font_size)
    
    if save_plot:
        plt.savefig(redshift_fig_folder + "hist_meanstddev_" + str(t) + ".png", 
                bbox_inches='tight') 
    
    if show_plot:
        plt.show() 
        
    plt.close()
    
    
    
def grad_norm_plot(grad_norm_D, 
                   grad_norm_G, 
                   redshift_fig_folder, 
                   t, 
                   save_plot, 
                   show_plot):
    
    plt.figure(figsize = (12,6))
    plt.title("grad_norms - if they are over 100 things are screwing up")
    plt.yscale('log')
    plt.plot(grad_norm_D, 
             color = "red", 
             label = "grad_norm_D",
             linewidth=0.5)
    plt.plot(grad_norm_G, 
             color = "blue", 
             label = "grad_norm_G",
             linewidth=0.5)
    plt.legend()
    
    if save_plot:
        plt.savefig(redshift_fig_folder + 'grad_norms_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show() 
        
    plt.close()
    
    
def grad_pen_plot(grad_pen,
                           redshift_fig_folder, 
                           t, 
                           save_plot, 
                           show_plot):
        
    plt.figure(figsize = (12,6))
    plt.title("Gradient Penalty")
#     plt.yscale('log')
    plt.plot(grad_pen, 
             color = "red",
             linewidth=0.5)
#     plt.legend()
    
    if save_plot:
        plt.savefig(redshift_fig_folder + 'grad_penalty_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show() 
        
    plt.close()
    
    
def encoding_mean_plot(f_enc_X_D_mean, 
                       f_enc_Y_D_mean, 
                       redshift_fig_folder, 
                       t, 
                       save_plot, 
                       show_plot):
    
    plt.figure(figsize = (12,6))
    plt.title("Embedding Means")
#     plt.yscale('log')
    plt.plot(f_enc_X_D_mean, 
             color = "red", 
             label = "f_enc_X_D_mean",
             linewidth=0.5)
    plt.plot(f_enc_Y_D_mean, 
             color = "blue", 
             label = "f_enc_Y_D_mean",
             linewidth=0.5)
    plt.plot([f_enc_X_D_mean[i] - f_enc_Y_D_mean[i] for i in range(len(f_enc_Y_D_mean))], 
             color = "black", 
             label = "mean_differences",
             linewidth=0.5)
    plt.legend()
    
    if save_plot:
        plt.savefig(redshift_fig_folder + 'embedding_means_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show() 
        
    plt.close()
    
    
def encoding_hist_plot_multiple(f_enc_X_D, 
                                f_enc_Y_D, 
                                log_plot, 
                                redshift_fig_folder, 
                                t, 
                                save_plot, 
                                show_plot):
    """
    Args:
        real(): real data
        epoch(integer): epoch number
        file_name(string): name of the file
        hd (integer) : if 0 it's a histogram, if 1 it's a pdf
        
    """
    
#     plt.figure(figsize = (20,10))
    plt.figure(figsize = (12,6))
    plot_min = min(f_enc_X_D.min(),f_enc_Y_D.min())
    plot_max = max(f_enc_X_D.max(),f_enc_Y_D.max())
#     print("plot_min = " + str(plot_min))
#     print("plot_max = " + str(plot_max))
    plt.xlim(plot_min-0.01,plot_max+0.01)
#     plt.xlim(-0.01, 0.01)
    plt.ylim(0, 50)
    font_size = 12
    log_plot = False
    
#     bins = np.linspace(-0.01, 0.01, 25)
    bins = np.linspace(plot_min-0.01,plot_max+0.01,100)
    
#     plt.yscale('log')
#     plt.xscale('log')

    plot_pdf = False
    
    batch_size = f_enc_X_D.shape[0]
    
    for m in range(min(batch_size,16)):
        
        f_enc_X_plot = f_enc_X_D[m].flatten()
        f_enc_Y_plot = f_enc_Y_D[m].flatten()
        
#         print("f_enc_X_plot = " + str(f_enc_X_plot))
#         print("f_enc_Y_plot = " + str(f_enc_Y_plot))
        
        if m==0:
#             print("plotting " + str(m))
            plt.hist(f_enc_X_plot, bins = bins, 
                     color = "b" ,alpha = 0.1, 
                     density=plot_pdf, label='f_enc_X_D')
            plt.hist(f_enc_Y_plot, bins = bins, 
                     color = "r" , alpha= 0.1, 
                     density=plot_pdf, label='f_enc_Y_D')
        else:
#             print("plotting " + str(m))
            plt.hist(f_enc_X_plot, bins = bins, 
                     color = "b" ,  alpha = 0.1, 
                     density=plot_pdf)
            plt.hist(f_enc_Y_plot, bins = bins, 
                     color = "r" , alpha= 0.1, 
                     density=plot_pdf)
            
            
#     plt.tick_params(axis='both', labelsize=font_size)
    plt.title('Encodings of Real and Generated Samples', fontsize=font_size)
    plt.legend(fontsize=font_size)
    
    if save_plot:
        plt.savefig(redshift_fig_folder + "hist_encodings_" + str(t) + ".png", 
                bbox_inches='tight') 
    
    if show_plot:
        plt.show() 
        
    plt.close()
        

def rbf_ratio_plot(loss_D_list, 
                   loss_G_list, 
                   var_est_D_list, 
                   var_est_G_list, 
                       redshift_fig_folder, 
                       t, 
                       save_plot, 
                       show_plot):
    
    plt.figure(figsize = (12,6))
    plt.title("Losses and Variance Estimates")
#     plt.yscale('log')
    plt.plot(loss_D_list, 
             color = "red", 
             label = "loss_D",
             linewidth=0.2)
    plt.plot(loss_G_list, 
             color = "blue", 
             label = "loss_G",
             linewidth=0.2)
    plt.plot(var_est_D_list, 
             color = "darkred", 
             label = "var_est_D",
             linewidth=0.2)
    plt.plot(var_est_G_list, 
             color = "darkblue", 
             label = "var_est_G",
             linewidth=0.2)
    plt.legend()
    
    if save_plot:
        plt.savefig(redshift_fig_folder + 'rbf_ratio_lossvar_' + str(t) + '.png', 
                    bbox_inches='tight')
    if show_plot:
        plt.show() 
        
    plt.close()
    
    
def mmd_contributions(k_xx_contrib,
                        k_yy_contrib,
                        k_xy_contrib,
                         D_or_G,
                             save_plot,
                             show_plot,
                             redshift_fig_folder,
                             t):
    """
    All the input the data should be inverse transformed to make it comparable with
    other transformation types.
    """              
    plt.figure(figsize = (12,6))
    plt.title("MMD Contributions by K(x,x), K(y,y) and K(x,y)")
    plt.xlabel("Epochs")
    plt.plot(k_xx_contrib, 
             label = "K(x,x)", 
             alpha = 0.5,
             color = "blue")
    plt.plot(k_yy_contrib, 
             label = "K(y,y)", 
             alpha = 0.5,
             color = "red")
    plt.plot(k_xy_contrib, 
             label = "K(x,y)", 
             alpha = 0.5,
             color = "green")
    plt.legend()
    if save_plot:
        if D_or_G == "D":
            plt.savefig(redshift_fig_folder + 'mmd_contributions_D_' + str(t) + '.png', 
                        bbox_inches='tight')
        else:
            plt.savefig(redshift_fig_folder + 'mmd_contributions_G_' + str(t) + '.png', 
                        bbox_inches='tight')
    if show_plot:
        plt.show() 
    
    plt.close()
    
    
    
    