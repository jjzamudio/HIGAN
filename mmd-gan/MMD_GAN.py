#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import graphviz
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable, grad
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.distributions as tdist
import h5py
import timeit
import time
import json
from scipy import stats
import pickle as pkl
from os import listdir
from os.path import isfile, join
import re
import shutil


# In[2]:


run_in_jupyter = False
try:
    cfg = get_ipython().config 
    run_in_jupyter = True
except:
    run_in_jupyter = False
    pass

if run_in_jupyter:
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
else: 
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
print("Run in Jupyter = " + str(run_in_jupyter))


# In[3]:


# DONT MOVE THESE TO THE UPPER CELL!!!!!
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


# ## Notes

# In[4]:


notes = "added Repulsive loss author's suggestions. disabled gradient penalty & unbiased MMD"


# ### Training Options

# In[5]:


run_mode = "training"                       # training OR continue = continue if load another model is loaded and continued to train
continue_train_folder = ""    # "jupyter-output-552" # the folder where the previous run is for the saved model to be trained further
netD_iter_file = ""         # netD_iter_xx.pth file that contains the state dict under models/
netG_iter_file = ""         # netG_iter_xx.pth file that contains the state dict under models/
optD_iter_file = ""    # optD_iter_10.pth
optG_iter_file = ""    # optG_iter_10.pth

batch_size = 32                       # BATCH_SIZE: batch size for training
gpu_device = 0                              # GPU_DEVICE: gpu id (default 0)
nc = 1                # NC: number of channels in images
cube_size = 64       # for our dataset more like one edge of the subcube
disc_lr = 5e-4               # LR: learning rate - default: 5e-5 (rmsprop) , 1e-4:adam
gen_lr = 1e-4               # LR: learning rate - default: 5e-5 (rmsprop) , 1e-4:adam
max_iter = 150         # MAX_ITER: max iteration for training

optimizer_choice = "adam"     # adam or rmsprop
adam_beta1 = 0.5     # default: 0.9    # coefficients used for computing running averages of gradient and its square 
adam_beta2 = 0.999     # default: 0.999
lr_decay = False               # Square root decay-> Just True or False        ||||Enter False or if True -> a numeric value
save_optim = False

manual_seed = 1
sample_size_multiplier = 101 * 6
n_samples = batch_size * sample_size_multiplier      # on prince, number of samples to get from the training cube
Diter_1 = 1   # default: 100
Giter_1 = 1      # default: 1
Diter_2 = 1      # default: 5
Giter_2 = 1      # default: 1
if run_mode == "continue":
    gen_iterations_limit = 1  
else:
    gen_iterations_limit = 1 # default = 25
edge_sample = cube_size
edge_test = 512

mmd2_D_train_limit = False       # if True, if MMD2_D is less than 0, the generator training will be skipped
mmd2_D_skip_G_train = False

enable_gradient_penalty = False  # Repulsive loss -> GP = False because there is spectral norm
lambda_gradpen = 1               # WGAN-GP = 10 | Demystifying MMD GANs = 1


# In[6]:


assert n_samples > Diter_1, "The gen_iterations wont work properly!"


# ### Model Options

# In[7]:


model_choice = "conv"                  # conv or con_fc
dimension_choice = "3D"                # 2D or 3D
nz = 64                          # hidden dimension channel

reconstruction_loss = False            # enable reconstruction loss or not
dist_ae = 'L2'                         # "L2" or "L1" -> Autoencoder reconstructruced cube loss choice,  "cos" doesnt work

repulsive_loss = 1.0                   # False | lambda_repulsive ex: {False, 0.1, 0.7, 1.0}
bounded_rbf_kernel = True             # True or False

mmd_kernel = "rbf"                     # rbf , rbf_ratio, poly , linear, rq , rq_linear
sigma_list = [1]         # sigma for RBF Kernel MMD
alpha_list = [2,5,10,20,40,80]         # alpha list for RQ Kernel MMD #[1, 2, 5, 10, 20, 40, 80]    # [0.2,0.5,1.0,2.0,5.0]
biased_mmd = False                     # author of Repulsive loss suggested False

weight_clip_enabled = False
left_clamp =  -0.01                    # default: -0.01
right_clamp = 0.01                     # default: 0.01

encoder_py = ""
decoder_py = ""

model_param_init = "normal"    # normal OR xavier (doesn't work right now)

"""
The explanations can be found in Ratio Matching MMD Nets (2018) in 
Equation 3.
"""
lambda_MMD = 1.0   # not used anywhere
lambda_AE_X = 0.0   
lambda_AE_Y = 0.0  
lambda_rg = 0.0 #16.0   # errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD \
                            # also for errG too

if not reconstruction_loss:
    lambda_AE_X = 0.0  
    lambda_AE_Y = 0.0  
    

min_var_est = 1e-30 # 1e-30, default:1e-8


# In[8]:


# if run_in_jupyter:
#     %run utils/power_spectrum_utils.py
# else:
from utils.power_spectrum_utils import *


# ### Plotting Options

# In[9]:


viz_multiplier = 1e2                # the norm multiplier in the 3D visualization
scatter_size_magnitude = False      # change scatter point radius based on the value of the point 
plot_3d_cubes = False
if run_in_jupyter:
    plot_show_3d = False             # shows the 3d scatter plot
    plot_show_other = True
    plot_save_3d = True             # whether to save or not as png 
    plot_save_other = True
else:
    plot_show_3d = False            # shows the 3d scatter plot
    plot_show_other = False
    plot_save_3d = True             # whether to save or not as png 
    plot_save_other = True


# ### Saving Options

# In[10]:


# if run_in_jupyter:
#     %run utils/logging_utils.py
# else:
from utils.logging_utils import *


# In[11]:


root_dir = "./"  # this goes to 
data_dir = "../"
# data_dir = "samples-2/"
new_output_folder = get_output_folder(run_in_jupyter = run_in_jupyter)
# new_output_folder = "drive-output-XX"   # for batch processing
experiment = root_dir + "outputs/" + new_output_folder + "/"       # : output directory of saved models
# print(experiment)

model_save_folder = experiment + "model/"
redshift_fig_folder = experiment + "figures/"        # folder to save mmd & related plots
redshift_3dfig_folder = experiment + "3d_figures/"   # folder to save 3D plots
testing_folder = experiment + "testing/"   # folder to save 3D plots

save_model_every = 10               # (every x epoch) frequency to save the model


# ### Dataset Options

# In[12]:


workers = 0        # WORKERS: number of threads to load data
redshift_info_folder = root_dir + "redshift_info/"   # save some info here as pickle to speed up processing
redshift_raw_file = "fields_z=5.0.hdf5"
# redshift_file = "redshift0_4th_root.h5"    # redshift cube to be used
                                        # standardized_no_shift_redshift0.h5
                                        # minmax_scale_01_redshift0.h5
                                        # minmax_scale_neg11_redshift0.h5
                                        # redshift0_4th_root.h5
                                        # redshift0_6th_root.h5
                                        # redshift0_8th_root.h5
                                        # redshift0_16th_root.h5
                                        # redshift0_4th_root_neg11.h5
root = 0 # should be an integer
inverse_transform = "log_scale_neg11"    # scale_01 / scale_neg11 / root / 
                                # root_scale_01 / root_scale_neg11
                                # log_scale_01 / log_scale_neg11
        



# create trial folder if it doesn't exist
if Path(experiment).exists() == False:
    print("Creating the output folder: {}".format(experiment))
    os.mkdir(experiment)


param_dict = {}
param_dict["notes"] = notes
param_dict["run_mode"] = run_mode
param_dict["continue_train_folder"] = continue_train_folder
param_dict["netD_iter_file"] = netD_iter_file
param_dict["netG_iter_file"] = netG_iter_file
param_dict["batch_size"] = batch_size
param_dict["gpu_device"] = gpu_device
param_dict["nc"] = nc
param_dict["cube_size"] = cube_size
param_dict["lr"] = "disc_lr:{}, gen_lr: {}".format(disc_lr,gen_lr)
param_dict["lr_decay"] = lr_decay
param_dict["max_iter"] = max_iter
param_dict["optimizer_choice"] = optimizer_choice
param_dict["manual_seed"] = manual_seed
param_dict["sample_size_multiplier"] = sample_size_multiplier
param_dict["n_samples"] = n_samples
param_dict["Diter_1"] = Diter_1
param_dict["Giter_1"] = Giter_1
param_dict["Diter_2"] = Diter_2
param_dict["Giter_2"] = Giter_2
param_dict["gen_iterations_limit"] = gen_iterations_limit
param_dict["edge_test"] = edge_test
param_dict["enable_gradient_penalty"] = enable_gradient_penalty
param_dict["lambda_gradpen"] = lambda_gradpen
param_dict["adam_beta1"] = adam_beta1
param_dict["adam_beta2"] = adam_beta2
param_dict["model_choice"] = model_choice
param_dict["dimension_choice"] = dimension_choice
param_dict["reconstruction_loss"] = reconstruction_loss
param_dict["dist_ae"] = dist_ae
param_dict["repulsive_loss"] = repulsive_loss
param_dict["mmd_kernel"] = mmd_kernel
param_dict["biased_mmd"] = biased_mmd
param_dict["bounded_rbf_kernel"] = bounded_rbf_kernel
param_dict["nz"] = nz
param_dict["model_param_init"] = model_param_init
param_dict["redshift_raw_file"] = redshift_raw_file
param_dict["root"] = root
param_dict["inverse_transform"] = inverse_transform
param_dict["lambda_MMD"] = lambda_MMD
param_dict["lambda_AE_X"] = lambda_AE_X
param_dict["lambda_AE_Y"] = lambda_AE_Y
param_dict["lambda_rg"] = lambda_rg
param_dict["sigma_list"] = sigma_list
param_dict["alpha_list"] = alpha_list
param_dict["min_var_est"] = min_var_est



print("Hyperparameter dictionary: {}".format(param_dict))

# with open(experiment+'hyperparam_dict.pickle', 'wb') as handle:
#     print("Saving the hyperparameters to {}".format(experiment))
#     pkl.dump(param_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
with open(experiment+'hyperparam_dict.json', 'w') as fp:
    print("Saving the hyperparameters to {}".format(experiment+'hyperparam_dict.json'))
    json.dump(param_dict, fp, indent=4)


# ## Redshift Data Load
# 
# Loading the raw data instead of the transformed data because the transformations are going to be done on the fly.

# In[ ]:


# f = h5py.File(data_dir + redshift_file, 'r')
f = h5py.File(data_dir + redshift_raw_file, 'r')
print("File used for analysis = " + str(f.filename))
f = f['delta_HI']


# ## Redshift Info Load

# In[ ]:


# create redshift info folder if it doesn't exist
if Path(redshift_info_folder).exists() == False:
    os.mkdir(redshift_info_folder)


# In[ ]:


# if run_in_jupyter:
#     %run utils/data_utils.py
# else:
from utils.data_utils import *


# In[ ]:


# min_cube,max_cube,mean_cube,stddev_cube = get_stats_cube(redshift_info_folder = redshift_info_folder,
#                                            redshift_file = redshift_file,
#                                            data_dir = data_dir)
min_cube,max_cube,mean_cube,stddev_cube = get_stats_cube(redshift_info_folder = redshift_info_folder,
                                           redshift_file = redshift_raw_file,
                                           data_dir = data_dir)

min_raw_cube,max_raw_cube,mean_raw_cube,stddev_raw_cube = get_stats_cube(redshift_info_folder = redshift_info_folder,
                                           redshift_file = redshift_raw_file,
                                           data_dir = data_dir)
print("\nTransformed  Data Summary Statistics:")
# print("File = " + str(redshift_file))
print("Min of data = " + str(min_cube))
print("Max of data = " + str(max_cube))
print("Mean of data = " + str(mean_cube))
print("Stddev of data = " + str(stddev_cube))

print("\nRaw Data Summary Statistics:")
print("File = " + str(redshift_raw_file))
print("Min of raw data = " + str(min_raw_cube))
print("Max of raw data = " + str(max_raw_cube))
print("Mean of raw data = " + str(mean_raw_cube))
print("Stddev of raw data = " + str(stddev_raw_cube))


# ## Figures Handling

# In[ ]:


# create figures folder if it doesn't exist
if Path(redshift_fig_folder).exists() == False:
    os.mkdir(redshift_fig_folder)
if Path(redshift_3dfig_folder).exists() == False:
    os.mkdir(redshift_3dfig_folder)


# ## 3D Plot

# In[ ]:


# if run_in_jupyter:
#     %run utils/plot_utils.py
# else:
from utils.plot_utils import *


# ## Data Loader

# In[ ]:


# from dataset import *
from dataset_2 import *
# from dataset_3 import *


# In[ ]:


# this has dataset loader in it so dont run this

# if run_in_jupyter:
#     %run test_3d_plot.py
# else:
# from test_3d_plot import *


# ## Dataset & DataLoader

# In[ ]:


# on prince
sampled_subcubes = HydrogenDataset(h5_file=redshift_raw_file,
                                    root_dir = data_dir,
                                    f = h5py.File(data_dir + redshift_raw_file, 'r')["delta_HI"],
                                    s_test = edge_test, 
                                    s_train = edge_sample,
                                    s_sample = edge_sample, 
                                    nsamples = n_samples,
                                   min_cube = min_cube,
                                  max_cube = max_cube,
                                  mean_cube = mean_cube,
                                  stddev_cube = stddev_cube,
                                   min_raw_cube = min_raw_cube,
                                  max_raw_cube = max_raw_cube,
                                  mean_raw_cube = mean_raw_cube,
                                  stddev_raw_cube = stddev_raw_cube,
                                  rotate_cubes = True,
                                  transform = inverse_transform,
                                  root = root,
                                  dimensions = dimension_choice)


# In[ ]:


sampled_subcubes[0]


# In[ ]:


# Get data
trn_loader = torch.utils.data.DataLoader(sampled_subcubes, 
                                         batch_size = batch_size,
                                         shuffle=True, 
                                         num_workers=int(workers))


# ## Checking 3D Plots

# In[ ]:


# # # dont run this in batch
# if run_in_jupyter:
#     test_3d_plot(edge_test = edge_test, 
#                  edge_sample = edge_sample,
#                  f = h5py.File(data_dir + redshift_file, 'r')["delta_HI"], 
#                  scatter_size_magnitude = scatter_size_magnitude,
#                  viz_multiplier = viz_multiplier,
#                  plot_save_3d = plot_save_3d,
#                  inverse_transform = inverse_transform,
#                  sampled_subcubes = sampled_subcubes)


# ## Model

# In[ ]:


# if run_in_jupyter:
#     %run utils/mmd_utils.py
#     %run utils/model_utils.py
#     %run utils/conv_utils.py

# else:
from utils.mmd_utils import *
from utils.model_utils import *
from utils.conv_utils import *


# Load & copy the decoder and encoder files to output folder for easier loading of architectures when resuming training:

# In[ ]:


# if run_in_jupyter:
#     %run models/encoder_v05_UpsampleConv_SpecNorm_Linear_2D.py
#     %run models/decoder_v05_UpsampleConv_Linear_2D.py
# else:

# if run_mode == "training":
from models.encoder_v06_UpsampleConv_SpecNorm_Linear_3D import *
from models.decoder_v06_UpsampleConv_Linear_3D import *

shutil.copy("models/encoder_v06_UpsampleConv_SpecNorm_Linear_3D.py",experiment)
shutil.copy("models/decoder_v06_UpsampleConv_Linear_3D.py",experiment)
# elif run_mode == "continue":
#     from outputs.
# else:
#     raise NotImplementedError


# In[ ]:


# if run_in_jupyter:
#     %run models/NetD.py
#     %run models/NetG.py
# else:
# from models.NetD import *
from models.NetD_NoDec import *
from models.NetG import *


# In[ ]:


# if run_in_jupyter:
#     %run one_sided.py
# else:
from one_sided import *


# In[ ]:


# if args.experiment is None:
#     args.experiment = 'samples'
# os.system('mkdir {0}'.format(args.experiment))

if model_save_folder is None:
    model_save_folder = 'samples'
os.system('mkdir {0}'.format(model_save_folder))


# In[ ]:


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# torch.cuda.max_memory_allocated(device=None)
# torch.cuda.max_memory_cached(device=None)
# torch.cuda.memory_allocated(device=None)
# torch.cuda.memory_cached(device=None)


# In[ ]:


if torch.cuda.is_available():
#     args.cuda = True
    cuda = True
#     torch.cuda.set_device(args.gpu_device)
    torch.cuda.set_device(gpu_device)
    print("Using GPU device", torch.cuda.current_device())
    print("Number of GPU: {}".format(torch.cuda.device_count()))
else:
    raise EnvironmentError("GPU device not available!")


# In[ ]:


# np.random.seed(seed=manual_seed)
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)
# torch.cuda.manual_seed(manual_seed)
cudnn.benchmark = True


# In[ ]:


print("\nDiscriminator")
D_encoder = Encoder(embed_channels = nz)
# D_decoder = Decoder(embed_channels = nz,
#                     D_encoder = D_encoder)
print("\nGenerator")
G_decoder = Decoder(embed_channels = nz,
                    D_encoder = D_encoder)


# In[ ]:


# netD = NetD(D_encoder, D_decoder, recon_loss = reconstruction_loss)
# netD = NetD(D_encoder, D_decoder)
netD = NetD(D_encoder)
# print("type netD: ", type(netD))
print("netD:", netD)


# In[ ]:


netG = NetG(G_decoder)
print("netG:", netG)


# In[ ]:


one_sided = ONE_SIDED()
print("oneSide:", one_sided)


# Save the models to be used when continuing the training:

# In[ ]:


def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
#             if isinstance(param, Parameter):
#                 # backwards compatibility for serialized parameters
#                 param = param.data
            own_state[name].copy_(param)


# In[ ]:


if run_mode == "continue":
    print("Loading saved models and parameters from file...")
    
    if netD_iter_file != "":
        print("Loading the Discriminator")
        load_my_state_dict(self = netD, 
                           state_dict = torch.load(f = root_dir + "outputs/" + continue_train_folder + "/model/" + netD_iter_file))
    if netG_iter_file != "":
        print("Loading the Generator")
        load_my_state_dict(self = netG, 
                           state_dict = torch.load(f = root_dir + "outputs/" + continue_train_folder + "/model/" + netG_iter_file))
    
    print("Loading complete...")


# In[ ]:


# if run_mode == "continue":
#     print("Loading saved models and parameters from file...")
    
#     if netD_iter_file != "":
#         print("Loading the Discriminator")
#         netD.load_state_dict(state_dict = torch.load(f = root_dir + "outputs/" + continue_train_folder + "/model/" + netD_iter_file))
#     if netG_iter_file != "":
#         print("Loading the Generator")
#         netG.load_state_dict(state_dict = torch.load(f = root_dir + "outputs/" + continue_train_folder + "/model/" + netG_iter_file))
    
#     print("Loading complete...")


# #### Network Visualization

# In[ ]:


# if run_in_jupyter:
#     %run utils/network_viz.py
# else:
from utils.network_viz import *


# In[ ]:


# dict(netD.named_parameters())


# In[ ]:


if run_in_jupyter:
    if dimension_choice == "2D":
        x = torch.randn(1,1,cube_size,cube_size).requires_grad_(True)
    elif dimension_choice == "3D":
        x = torch.randn(1,1,cube_size,cube_size,cube_size).requires_grad_(True)
    else:
        raise NotImplementedError
    y = netD.encoder(Variable(x))
    g = make_dot(y,
             params=dict(list(netD.encoder.named_parameters()) + [('x', x)]))
    g.view(directory=experiment, filename="netD_encoder_viz")

    z = netG.decoder(Variable(y))
    g = make_dot(z,
             params=dict(list(netG.decoder.named_parameters()) + [('z', z)]))
    g.view(directory=experiment, filename="netG_decoder_viz")


# #### Weights Initialization

# In[ ]:


if run_mode != "continue":
    print("initializing parameters...")
    if netD_iter_file == "":
        print("Initialize Discriminator Parameters")
        netD.apply(lambda x: weights_init(x,init_type = model_param_init))
    if netG_iter_file == "":
        print("Initialize Generator Parameters")
        netG.apply(lambda x: weights_init(x,init_type = model_param_init))
    one_sided.apply(lambda x: weights_init(x,init_type = model_param_init))
    


# In[ ]:



"""
see the parameters of the networks

The convolutional kernels:
torch.Size([2, 1, 4, 4, 4])

What are these for?
torch.Size([4])
"""
print("Discriminator Encoder:")
for p in netD.encoder.parameters():
    print(p.shape)
    
# if reconstruction_loss:
# print("\nDiscriminator Decoder:")  
# for p in netD.decoder.parameters():
#     print(p.shape)
    
print("\nGenerator Decoder:")  
for p in netG.decoder.parameters():
    print(p.shape)
    
# for name, param in netD.encoder.named_parameters():
#     if param.requires_grad:
#         print(str(name) + str(param.shape) + str(param.data))


# In[ ]:


# put variable into cuda device


"""
errD.backward(mone)
optimizerD.step()

errG.backward(one)
optimizerG.step()
"""
one = torch.tensor(1.0).cuda()
#one = torch.cuda.FloatTensor([1])
mone = one * -1


# In[ ]:


if cuda:
    netG.cuda()
    netD.cuda()
    one_sided.cuda()


# #### Optimizer Choice

# In[ ]:


# if run_mode == "training":
if optimizer_choice == "rmsprop":
    optimizerG = torch.optim.RMSprop(netG.parameters(), 
                                     lr=gen_lr)
    optimizerD = torch.optim.RMSprop(netD.parameters(), 
                                     lr=disc_lr)
elif optimizer_choice == "adam":
    optimizerG = torch.optim.Adam(netG.parameters(), 
                                     lr=gen_lr,
                                  betas = (adam_beta1, adam_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), 
                                     lr=disc_lr,
                                  betas = (adam_beta1, adam_beta2))
elif run_mode == "continue":
    if optG_iter_file != "":
        optimizerG.load_state_dict(state_dict = torch.load(f = root_dir + "outputs/" + continue_train_folder + "/model/" + optG_iter_file))
    if optD_iter_file != "":
        optimizerD.load_state_dict(state_dict = torch.load(f = root_dir + "outputs/" + continue_train_folder + "/model/" + optD_iter_file))

else:
    raise NotImplementedError 
    

    


# In[ ]:


# https://github.com/pytorch/pytorch/issues/4632
torch.backends.cudnn.benchmark = False


# ### GPU Used

# In[ ]:


# # delete variable
# del x
# # clear memory cache
# torch.cuda.empty_cache()


# In[ ]:


def pretty_size(size):
	"""Pretty prints a torch.Size object"""
	assert(isinstance(size, torch.Size))
	return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__, 
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
												   type(obj.data).__name__, 
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "", 
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass        
	print("Total size:", total_size)
    
dump_tensors(gpu_only=True)


# In[ ]:


def check_gpu(run_in_jupyter):
    if run_in_jupyter:
        get_ipython().system('nvidia-smi')
    else:
        import subprocess
        print(subprocess.check_output(['nvidia-smi']))

check_gpu(run_in_jupyter)


# ## Training Loop

# In[ ]:


time_loop = timeit.default_timer()
print("time = " + str(time_loop))

time_1_list = []
time_2_list = []

gen_iterations = 1  # the code default is = 0

# lists for tracking - Discriminator side
mmd2_D_before_ReLU_list = []
mmd2_D_after_ReLU_list = []
one_side_errD_list = []
L2_AE_X_D_list = []
L2_AE_Y_D_list = []
errD_list = []
loss_D_list = []
var_est_D_list = []
plotted_mmdD = False

# lists for tracking - Generator side
mmd2_G_before_ReLU_list = []
mmd2_G_after_ReLU_list = []
one_side_errG_list = []
errG_list = []
loss_G_list = []
var_est_G_list = []
plotted_mmdG = False

# lists for tracking count of nonzero voxels
log_nonzero_recon_over_real_list = []

# list for tracking gradient norms for generator and discriminator
grad_norm_D = []
grad_norm_G = []
grad_norm_pen = []

# lists for tracking the sum of all cubes in a minibatch
sum_noise_gen = []
sum_noise_gen_recon = []
sum_real = []
sum_real_recon = []

# embedding means
f_enc_X_D_mean = []
f_enc_Y_D_mean = []

# MMD contribution tracking
k_xx_contrib_D = []
k_yy_contrib_D = []
k_xy_contrib_D = []
k_xx_contrib_G = []
k_yy_contrib_G = []
k_xy_contrib_G = []

# Avg Zero per Cube Tracking
zero_count_real = []
zero_count_gen = []






fixed_noise_set = 0

for t in range(max_iter):
    print("\n-----------------------------------------------")
    print("Epoch = " + str(t+1) + " / " + str(max_iter))
    print("----------------------------------------------- \n")
    
    data_iter = iter(trn_loader)
    print("len(trn_loader) = " + str(len(trn_loader)))
    i = 0
    plotted = 0
    plotted_2 = 0
    plotted_3 = 0
    plotted_4 = 0   # grad norm plotting controller
    
    
    #Learning rate decay
    if (t+1) % 5 == 0 and lr_decay == True:
        # Linear Decay
        optimizerD.param_groups[0]['lr'] /= lr_decay
        optimizerG.param_groups[0]['lr'] /= lr_decay
        
        # Square Root Decay
#         optimizerD.param_groups[0]['lr'] = np.sqrt(optimizerD.param_groups[0]['lr'])
#         optimizerG.param_groups[0]['lr'] = np.sqrt(optimizerG.param_groups[0]['lr'])

        print("Learning rate changed to - D: {}, G: {}".format(optimizerD.param_groups[0]['lr'],
                                                               optimizerG.param_groups[0]['lr']))
    
    
    
    
    while (i < len(trn_loader)):
        
        # check gpu memory usage
        if plotted < 1:
            check_gpu(run_in_jupyter)
        
        # ---------------------------
        #        Optimize over NetD
        # ---------------------------
        print("Optimize over NetD")
        for p in netD.parameters():
            p.requires_grad = True

            
        """
        What does the below if-else do?
        Trains the discriminator for a lot more when the training
        is starting, then switches to a more frequent generator
        training regime.
        """
        print("gen_iterations = " + str(gen_iterations))
        if gen_iterations < gen_iterations_limit or gen_iterations % 500 == 0:
            Diters = Diter_1
            Giters = Giter_1
        else:
            Diters = Diter_2
            Giters = Giter_2

        for j in range(Diters):
            if i == len(trn_loader):
                break

            time_1 = time.time()
            print("j / Diter = " + str(j+1) + " / " + str(Diters))
            
            start = time.time()

            if weight_clip_enabled == True:
                for p in netD.encoder.parameters():
                    p.data.clamp_(left_clamp, right_clamp)
            
            if True:

                i += 1

                netD.zero_grad()

                x = Variable(x.cuda().float())

                batch_size = x.size(0)

                
                f_enc_X_D, f_enc_X_size = netD(x)
                
                noise = torch.cuda.FloatTensor(f_enc_X_size).normal_(0, 1)
                with torch.no_grad():
                    y = netG(noise) # freeze netG
                f_enc_Y_D, _ = netD(y)

                # compute biased MMD2 and use ReLU to prevent negative value
                if mmd_kernel == "rbf":
                    mmd2_D, contrib_list = mix_rbf_mmd2(f_enc_X_D, 
                                          f_enc_Y_D, 
                                          sigma_list,
                                          biased=biased_mmd,
                                          repulsive_loss = repulsive_loss, 
                                          bounded_kernel = bounded_rbf_kernel)
                    k_xx_contrib_D.append(contrib_list[0])
                    k_yy_contrib_D.append(contrib_list[1])
                    k_xy_contrib_D.append(contrib_list[2])
                elif mmd_kernel == "rbf_ratio":
                    loss_D, mmd2_D, var_est_D = mix_rbf_mmd2_and_ratio(
                                            X = f_enc_X_D, 
                                           Y = f_enc_Y_D, 
                                           sigma_list = sigma_list, 
                                           biased=biased_mmd,
                                            min_var_est = min_var_est)
                    loss_D_list.append(loss_D.item())
                    var_est_D_list.append(var_est_D.item())
                    if plotted_mmdD == False:
                        print("mmd2_D = " + str(mmd2_D.item()))
                        print("loss_D = " + str(loss_D.item()))
                        print("var_est_D = " + str(var_est_D.item()))
                        plotted_mmdD = True
                elif mmd_kernel == "rq":
                    mmd2_D, contrib_list = mix_rq_mmd2(X = f_enc_X_D, 
                                           Y = f_enc_Y_D, 
                                           alpha_list = alpha_list, 
                                           biased=biased_mmd, 
                                           repulsive_loss = repulsive_loss)
                    k_xx_contrib_D.append(contrib_list[0])
                    k_yy_contrib_D.append(contrib_list[1])
                    k_xy_contrib_D.append(contrib_list[2])
                elif mmd_kernel == "rq_linear":
                    mmd2_D, contrib_list = mix_rq_mmd2(X = f_enc_X_D, 
                                           Y = f_enc_Y_D, 
                                           alpha_list = alpha_list, 
                                           biased=biased_mmd, 
                                           repulsive_loss = repulsive_loss)
                    mmd2_D = mmd2_D + linear_mmd2(f_of_X = f_enc_X_D, 
                                                    f_of_Y = f_enc_Y_D)
                    k_xx_contrib_D.append(contrib_list[0])
                    k_yy_contrib_D.append(contrib_list[1])
                    k_xy_contrib_D.append(contrib_list[2])
                elif mmd_kernel == "poly":
                    mmd2_D = poly_mmd2(f_enc_X_D, f_enc_Y_D)
                elif mmd_kernel == "linear":
                # linear MMD
                    mmd2_D = linear_mmd2(f_of_X = f_enc_X_D, 
                                         f_of_Y = f_enc_Y_D)


    #             print("mmd2_D before ReLU = " + str(mmd2_D.item()))
                mmd2_D_before_ReLU_list.append(mmd2_D.item())
                mmd2_D = F.relu(mmd2_D)
    #             print("mmd2_D after ReLU = " + str(mmd2_D.item()))
                mmd2_D_after_ReLU_list.append(mmd2_D.item())

                # compute rank hinge loss
                one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))
                one_side_errD_list.append(one_side_errD.item())

                # compute L2-loss of AE
                """
                Reconstruction Loss
                """
#               if reconstruction_loss:
#                 L2_AE_X_D = match(x.view(batch_size, -1), f_dec_X_D, dist_ae)
#                 L2_AE_Y_D = match(y.view(batch_size, -1), f_dec_Y_D, dist_ae)
#                 L2_AE_X_D_list.append(L2_AE_X_D.item())
#                 L2_AE_Y_D_list.append(L2_AE_Y_D.item())
    
                if not enable_gradient_penalty and reconstruction_loss:
                    """Original MMD GAN Loss"""
                    errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD  - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D 
                elif not enable_gradient_penalty and not reconstruction_loss:
                    errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD
                else:
                    gradnorm_D = calc_gradient_penalty_new(netD = netD, 
                                                       real_data = x, 
                                                       fake_data = y)
                    grad_norm_pen.append(gradnorm_D.item())
                    errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD - lambda_gradpen * gradnorm_D

                errD_list.append(errD.item())
                errD.backward(mone)
                optimizerD.step()

            # if fixed_noise_set == 0:
            
            #     fixed_noise = torch.cuda.FloatTensor(f_enc_X_size).normal_(0, 1)
            #     if model_choice == "conv_fc":
            #         fixed_noise = fixed_noise[0]  # plot just one cube
            #         fixed_noise = fixed_noise.view(1,-1)
            #     print("Fixed Noise size = " + str(fixed_noise.size()))
            #     fixed_noise = Variable(fixed_noise, 
            #                            requires_grad=False)
            #     fixed_noise_set = fixed_noise_set + 1
        
            # Plotting Discriminator Plots
            if j % 2 == 0 and plotted < 1:
                if True:
#                 try:
                    
    
                    """
                    Plotting Different Discriminator Related Values
                    """
                    print("\nPlotting Different Discriminator Related Values")
    
                    plot_list = [mmd2_D_before_ReLU_list,mmd2_D_after_ReLU_list,
                                 one_side_errD_list, L2_AE_X_D_list,
                                 L2_AE_Y_D_list, errD_list ]
                    plot_title_list = ["mmd2_D_before_ReLU_list", "mmd2_D_after_ReLU_list",
                                       "one_side_errD_list", "L2_AE_X_D_list",
                                       "L2_AE_Y_D_list", "errD_list - D loss goes to 0: failure mode"]
                    for plot_no in range(len(plot_list)):
                        mmd_loss_plots(fig_id = plot_no, 
                                        fig_title = plot_title_list[plot_no], 
                                        data = plot_list[plot_no], 
                                        show_plot = plot_show_other, 
                                        save_plot = plot_save_other, 
                                        redshift_fig_folder = redshift_fig_folder,
                                      t = t,
                                      dist_ae = dist_ae)


                if True:
                    noise_gen_cube = y.cpu().detach().numpy()
                    real_cube = x.cpu().detach().numpy()

                    histogram_mean_confint(noise = noise_gen_cube, 
                                real = real_cube, 
                                log_plot = False, 
                                redshift_fig_folder = redshift_fig_folder,
                                t = t,
                                save_plot = plot_save_other,
                                show_plot = plot_show_other)

                    noise_gen_cube = inverse_transform_func(cube = noise_gen_cube,
                                                  inverse_type = inverse_transform, 
                                             sampled_dataset = sampled_subcubes)
                    real_cube = inverse_transform_func(cube = real_cube,
                                                  inverse_type = inverse_transform, 
                                             sampled_dataset = sampled_subcubes)
                    print("\nInverse Transformed Subcubes:")
                    print("noise_gen_cube max = " + str(noise_gen_cube.max()) + ", min = " + str(noise_gen_cube.min())                         + ", mean = " + str(noise_gen_cube.mean()))
                    print("real_cube max = " + str(real_cube.max()) + ", min = " + str(real_cube.min())                         + ", mean = " + str(real_cube.mean()))
                    

                    sum_real.append(real_cube.sum())
                    sum_noise_gen.append(noise_gen_cube.sum())
                    
                    print("\nPlotting the sum of values across a minibatch")
                    plot_minibatch_value_sum(sum_real = sum_real,
                             sum_real_recon = sum_real_recon,
                             sum_noise_gen = sum_noise_gen,
                             sum_noise_gen_recon = sum_noise_gen_recon,
                             save_plot = plot_save_other,
                             show_plot = plot_show_other,
                             redshift_fig_folder = redshift_fig_folder,
                             t = t)
                    

                    """
                    # convert values smaller than 6.881349e-10 to zero
                    # this value is the minimum nonzero in the real cube                    
                    """
                    noise_gen_cube[noise_gen_cube < 6.881349e-10] == 0.0                    
                    
                    
                    
                    """
                    Power Spectrum Comparisons
                    NEEDS INVERSE-TRANSFORMED DATA
                    """
                    print("\nPower Spectrum Comparisons")
                    
                    with torch.no_grad():
                        plot_ps_percentiles_many(netG = netG,   # should be inverse_transformed
                                                 f_enc_X_size = f_enc_X_size,
                                                 total_cubes = 200,
                                                raw_cube_mean = sampled_subcubes.mean_val,    # mean of the whole raw data cube (fields=z0.0)
                                                inverse_transform = inverse_transform,
                                                 sampled_subcubes = sampled_subcubes,
                                                save_plot = plot_save_other,
                                        show_plot = plot_show_other,
                                        redshift_fig_folder = redshift_fig_folder,
                                        t = t,
                                        threads=1, 
                                        MAS="CIC", 
                                        axis=0, 
                                        BoxSize=75.0/2048*cube_size,
                                        data_dim = dimension_choice)
    
                    
                    """
                    2D Visualizations
                    NEEDS INVERSE TRANSFORMED + Transformed Log scale 01 data
                    """
                    log_01_scale = "log_scale_01"
                    noise_gen_cube = transform_func(cube = noise_gen_cube,
                                                   inverse_type = log_01_scale,
                                                   self = sampled_subcubes)
                    real_cube = transform_func(cube = real_cube,
                                                   inverse_type = log_01_scale,
                                                   self = sampled_subcubes)

                    visualize2d(real = real_cube, 
                                fake = noise_gen_cube, 
                                raw_cube_mean = sampled_subcubes.mean_raw_val, 
                                scaling = inverse_transform,
                                redshift_fig_folder = redshift_fig_folder,
                                t = t,
                                save_plot = plot_save_other, 
                                show_plot = plot_show_other)
                    
                    """
                    Count zeros
                    """
                    power_cube = 3 if dimension_choice == '3D' else 2
                    zero_count_real.append(np.count_nonzero(real_cube==0.0) / batch_size / cube_size**power_cube)
                    zero_count_gen.append(np.count_nonzero(noise_gen_cube==0.0) / batch_size / cube_size**power_cube)
                    zero_count_plot(zero_count_real = zero_count_real, 
                                       zero_count_gen = zero_count_gen, 
                                           redshift_fig_folder = redshift_fig_folder, 
                                           t = t, 
                                           save_plot = plot_save_other, 
                                           show_plot = plot_show_other)
                    
                plotted = plotted + 1

        print("Finished optimizing over NetD \n")


        # ---------------------------
        #        Optimize over NetG
        # ---------------------------
        print("Optimize over NetG")
        for p in netD.parameters():
            p.requires_grad = False

        print("Giters = " + str(Giters))
        for j in range(Giters):
            print("i = " + str(i))
            
            try:
                if mmd2_D.item() <= 0.0 and mmd2_D_skip_G_train == True:
                    print("Not training the generator because mmd2_D is less than 0")
                    
                    break
            except:
                pass
            
            print("len(trn_loader) = " + str(len(trn_loader)))
            if i == len(trn_loader):
                print("Breaking from the Generator training loop")
                break
                

            print("j / Giter = " + str(j+1) + " / " + str(Giters))
#             data = data_iter.next()
            x = data_iter.next()
            i += 1
            netG.zero_grad()

            x = Variable(x.cuda().float())
            batch_size = x.size(0)

            f_enc_X, f_enc_X_size = netD(x)

            noise = torch.cuda.FloatTensor(f_enc_X_size).normal_(0, 1)
            
            y = netG(noise)
            
            # convert values smaller than 6.881349e-10 to zero
            # this value is the minimum nonzero in the real cube
#             y[y < 6.881349e-10] == 0.0

            # output of the discriminator with noise input
            f_enc_Y, _ = netD(y)

            # compute biased MMD2 and use ReLU to prevent negative value
            if mmd_kernel == "rbf":
                mmd2_G, contrib_list = mix_rbf_mmd2(f_enc_X, 
                                      f_enc_Y, 
                                      sigma_list, 
                                      biased = biased_mmd,
                                      repulsive_loss = False, 
                                      bounded_kernel = False)
                k_xx_contrib_G.append(contrib_list[0])
                k_yy_contrib_G.append(contrib_list[1])
                k_xy_contrib_G.append(contrib_list[2])
            elif mmd_kernel == "rbf_ratio":
                loss_G, mmd2_G, var_est_G = mix_rbf_mmd2_and_ratio(
                                            X = f_enc_X, 
                                           Y = f_enc_Y, 
                                           sigma_list = sigma_list, 
                                           biased=biased_mmd,
                                            min_var_est = min_var_est)
                loss_G_list.append(loss_G.item())
                var_est_G_list.append(var_est_G.item())
                if plotted_mmdG == False:
                    print("mmd2_G = " + str(mmd2_G.item()))
                    print("loss_G = " + str(loss_G.item()))
                    print("var_est_G = " + str(var_est_G.item()))
                    plotted_mmdG = True
            elif mmd_kernel == "rq":
                mmd2_G, contrib_list = mix_rq_mmd2(X = f_enc_X, 
                                       Y = f_enc_Y, 
                                       alpha_list = alpha_list, 
                                       biased=True, 
                                       repulsive_loss = False)
                k_xx_contrib_G.append(contrib_list[0])
                k_yy_contrib_G.append(contrib_list[1])
                k_xy_contrib_G.append(contrib_list[2])
            elif mmd_kernel == "rq_linear":
                mmd2_G, contrib_list = mix_rq_mmd2(X = f_enc_X, 
                                       Y = f_enc_Y, 
                                       alpha_list = alpha_list, 
                                       biased=biased_mmd, 
                                       repulsive_loss = False)
                mmd2_G = mmd2_G + linear_mmd2(f_of_X = f_enc_X, 
                                                f_of_Y = f_enc_Y)
                k_xx_contrib_G.append(contrib_list[0])
                k_yy_contrib_G.append(contrib_list[1])
                k_xy_contrib_G.append(contrib_list[2])
            elif mmd_kernel == "poly":
                mmd2_G = poly_mmd2(f_enc_X, f_enc_Y)
            elif mmd_kernel == "linear":
                mmd2_G = linear_mmd2(f_enc_X, f_enc_Y)
    
            mmd2_G_before_ReLU_list.append(mmd2_G)
            mmd2_G = F.relu(mmd2_G)
            mmd2_G_after_ReLU_list.append(mmd2_G)

            # compute rank hinge loss
            one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))
            one_side_errG_list.append(one_side_errG)

            errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG 
#             errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG \
#                     + calc_gradient_penalty(x.data, y.data, lambda_gradpen)
            print("errG = " + str(errG.item()))
            errG_list.append(errG.item())
            
        
            errG.backward(one)
            optimizerG.step()
            
            if plotted_3 < 1:
                """
                Plotting Generator Related Values
                """
                plot_list = [mmd2_G_before_ReLU_list,mmd2_G_after_ReLU_list,
                             one_side_errG_list, errG_list ]
                plot_title_list = ["mmd2_G_before_ReLU_list", "mmd2_G_after_ReLU_list",
                                   "one_side_errG_list","errG_list"]
                for plot_no in range(len(plot_list)):
                    mmd_loss_plots(fig_id = plot_no, 
                                    fig_title = plot_title_list[plot_no], 
                                    data = plot_list[plot_no], 
                                    show_plot = plot_show_other, 
                                    save_plot = plot_save_other, 
                                    redshift_fig_folder = redshift_fig_folder,
                                  t = t,
                                  dist_ae = dist_ae)           
            
                plotted_3 = plotted_3 + 1
                
        # add one to gen_iterations
        gen_iterations += 1

        run_time = (timeit.default_timer() - time_loop) / 60.0
        print("run_time = " + str(run_time), flush = True)
        
        try:
            print('[%3d/%3d][%3d/%3d] [%5d] (%.2f m) MMD2_D %.10f MMD2_G %.10f hinge %.6f loss_D %.6f Loss_G %.6f f_enc_X_D.mean() %.6f f_enc_Y_D.mean() %.6f |gD| %.10f |gG| %.10f'
                  % (t, max_iter, i, len(trn_loader), gen_iterations, run_time,
                     mmd2_D_before_ReLU_list[-1], mmd2_G_before_ReLU_list[-1], one_side_errD.item(),
                     errD.item(), errG.item(),
                     f_enc_X_D.mean().item(), f_enc_Y_D.mean().item(),
                     grad_norm(netD), grad_norm(netG)))
        except:
            print("COULDNT PRINT WHOLE LOG")
            print("mmd2_D: {}".format(mmd2_D.item()))
            pass

        
        # plotting gradient norms for monitoring
        grad_normD = grad_norm(netD)
        grad_norm_D.append(grad_normD)
         
        try:
            grad_norm_G.append(grad_norm(netG))
        except:
            grad_norm_G.append(0.0)
            pass
        
        # embedding means
        f_enc_X_D_mean.append(f_enc_X_D.mean().item())
        f_enc_Y_D_mean.append(f_enc_Y_D.mean().item())
        
        if plotted_4 < 1:
            grad_norm_plot(grad_norm_D = grad_norm_D, 
                           grad_norm_G = grad_norm_G, 
                           redshift_fig_folder = redshift_fig_folder, 
                           t = t, 
                           save_plot = plot_save_other, 
                           show_plot = plot_show_other)
            
            grad_pen_plot(grad_pen = grad_norm_pen,
                           redshift_fig_folder = redshift_fig_folder, 
                           t = t, 
                           save_plot = plot_save_other, 
                           show_plot = plot_show_other)
            
            encoding_mean_plot(f_enc_X_D_mean = f_enc_X_D_mean, 
                               f_enc_Y_D_mean = f_enc_Y_D_mean, 
                               redshift_fig_folder = redshift_fig_folder, 
                               t = t, 
                               save_plot = plot_save_other, 
                               show_plot = plot_show_other)
            print("MMD_D Contribution Plots")
            mmd_contributions(k_xx_contrib = k_xx_contrib_D,
                        k_yy_contrib = k_yy_contrib_D,
                        k_xy_contrib = k_xy_contrib_D,
                         D_or_G = "D",
                             save_plot = plot_save_other,
                             show_plot = plot_show_other,
                             redshift_fig_folder = redshift_fig_folder,
                             t = t)
            print("MMD_G Contribution Plots")
            mmd_contributions(k_xx_contrib = k_xx_contrib_G,
                        k_yy_contrib = k_yy_contrib_G,
                        k_xy_contrib = k_xy_contrib_G,
                         D_or_G = "G",
                             save_plot = plot_save_other,
                             show_plot = plot_show_other,
                             redshift_fig_folder = redshift_fig_folder,
                             t = t)
            
            if mmd_kernel == "rbf_ratio":
                rbf_ratio_plot(loss_D_list = loss_D_list, 
                   loss_G_list = loss_G_list, 
                   var_est_D_list = var_est_D_list, 
                   var_est_G_list = var_est_G_list, 
                   redshift_fig_folder = redshift_fig_folder, 
                   t = t, 
                   save_plot = plot_save_other, 
                   show_plot = plot_show_other)
            
            
            plotted_4 = plotted_4 + 1


    if t % save_model_every == 0:
        # Saving the trained model
        print("Saving the model state_dict()")
        torch.save(netG.state_dict(), 
                   '{0}/netG_iter_{1}.pth'.format(model_save_folder, t))
        torch.save(netD.state_dict(), 
                   '{0}/netD_iter_{1}.pth'.format(model_save_folder, t))
        
        # Saving the state of the optimizer
#         torch.save('optimizer': optimizerG.state_dict(), 
#                    '{0}/optG_iter_{1}.pth'.format(model_save_folder, t))
        if save_optim:
            torch.save(optimizerG.state_dict(), 
                       '{0}/optG_iter_{1}.pth'.format(model_save_folder, t))
            torch.save(optimizerD.state_dict(), 
                       '{0}/optD_iter_{1}.pth'.format(model_save_folder, t))

print("TRAINING DONE!")

