# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:32:43 2018

@author: Juan Jose Zamudio
"""
import random
import numpy as np
import torch
import h5py
random.seed(a=1)
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable, grad

mean_5=14592.24
std_5=922711.56
max_5=4376932000
mask_value=10**2

mean_l5 = 2.7784111
std_l5 = 1.5777067
max_l5 =22.199614
max_l5 = 21.451267

mean_5_2d = 14280.155
std_5_2d = 89914.586
max_5_2d = 47676240

#Stats for log
mean_l5_2d = 4.9034805
std_l5_2d = 2.3593297
max_l5_2d = 17.636091


cuda=True


def define_test(s_test, s_train):
    #2048/16=128
    m=2048/s_train
    x=random.randint(0,m)*s_train
    y=random.randint(0,m)*s_train
    z=random.randint(0,m)*s_train
    #print(x,y,z)
    return {'x':[x,x+s_test], 'y':[y,y+s_test], 'z':[z,z+s_test]}

def check_coords(test_coords, train_coords):
    valid = True
    for i in ['x','y','z']:
        r=(max(test_coords[i][0], 
               train_coords[i][0]), 
           min(test_coords[i][1],
               train_coords[i][1]))
        if r[0]<=r[1]:
            valid = False
    return valid

def get_samples(file, s_sample, nsamples, test_coords):
    #n is size of minibatch, get valid samples (not intersecting with test_coords)
    sample_list=[]
    m=2048-s_sample
    for n in range(nsamples):
        #print("Sample No = " + str(n + 1) + " / " + str(nsamples))
        sample_valid=False
        while sample_valid==False:
            x = random.randint(0,m)
            y = random.randint(0,m)
            z = random.randint(0,m)
            sample_coords = {'x':[x,x+s_sample], 
                             'y':[y,y+s_sample], 
                             'z':[z,z+s_sample]}
            
            sample_valid = check_coords(test_coords, sample_coords)
        
    
        
        sample_list.append(sample_coords)
    
    #Load cube and get samples and convert them to np.arrays
    sample_array=[]
    #f file has to be opened outisde the function
    for c in sample_list:
        print(c)
        a = file[c['x'][0]:c['x'][1],
              c['y'][0]:c['y'][1],
              c['z'][0]:c['z'][1]]
        
        sample_array.append( np.array(a))
    
    return np.array(sample_array)


def get_max_cube(file):
    
    max_list = []
    for i in range(file.shape[0]):
        #print(np.max(f[i:i+1,:,:]))
        max_list.append(np.max(file[i:i+1,:,:]))
    max_cube = max(max_list)
   
    return max_cube

def get_min_cube(file):
    min_list = [] 
    for i in range(file.shape[0]):
        #print(np.max(f[i:i+1,:,:]))
        min_list.append(np.min(file[i:i+1,:,:]))
    min_cube = min(min_list)
    return min_cube

def get_mean_cube(file):
    mean_list = []
    for i in range(file.shape[0]):
        #print(np.max(f[i:i+1,:,:]))
        mean_list.append(np.mean(file[i:i+1,:,:]))
    mean_cube = np.mean(mean_list)
    return mean_cube

def get_stddev_cube(file, mean_cube):
    variance_list = []
    for i in range(file.shape[0]):
        variance_list.append(np.mean(np.square(file[i:i+1,:,:] - mean_cube)))
    stddev_cube = np.sqrt(np.mean(variance_list))
    return stddev_cube




##To calculate power spectrums

def power_spectrum_np(cube, mean_raw_cube, SubBoxSize):

    nc = cube.shape[2] # define how many cells your box has
    #nc = nc*nc*nc
    
    delta = cube/mean_raw_cube - 1.0

    # get P(k) field: explot fft of data that is only real, not complex
    delta_k = np.abs(np.fft.rfftn(delta)) 
    Pk_field =  delta_k**2

    # get 3d array of index integer distances to k = (0, 0, 0)
    dist = np.minimum(np.arange(nc), np.arange(nc,0,-1))
    dist_z = np.arange(nc//2+1)
    dist *= dist
    dist_z *= dist_z
    dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist_z)

    ################ NEW #################
    dist_3d  = np.ravel(dist_3d)
    Pk_field = np.ravel(Pk_field)
    
    k_bins = np.arange(nc//2+1)
    k      = 0.5*(k_bins[1:] + k_bins[:-1])*2.0*np.pi/SubBoxSize
    
    Pk     = np.histogram(dist_3d, bins=k_bins, weights=Pk_field)[0]
    Nmodes = np.histogram(dist_3d, bins=k_bins)[0]
    Pk     = (Pk/Nmodes)*(SubBoxSize/nc**2)**3
    
    k = k[1:];  Pk = Pk[1:]
    
    return k, Pk


def power_spectrum_np_2d(cube, mean_raw_cube, SubBoxSize):
    #print(cube.shape)
    nc = cube.shape[1] # define how many cells your box has
    delta = cube/mean_raw_cube - 1.0

    # get P(k) field: explot fft of data that is only real, not complex
    delta_k = np.abs(np.fft.rfftn(delta)) 
    Pk_field =  delta_k**2
    #print(Pk_field.shape)

    # get 3d array of index integer distances to k = (0, 0, 0)
    dist = np.minimum(np.arange(nc), np.arange(nc,0,-1))
    dist_z = np.arange(nc//2+1)
    dist *= dist
    dist_z *= dist_z
    dist_2d = np.sqrt(dist[:, None] + dist_z)

    ################ NEW #################
    dist_2d  = np.ravel(dist_2d)
    Pk_field = np.ravel(Pk_field)
    
    k_bins = np.arange(nc//2+1)
    k      = 0.5*(k_bins[1:] + k_bins[:-1])*2.0*np.pi/SubBoxSize
    
    #print(dist_2d.shape)
    Pk     = np.histogram(dist_2d, bins=k_bins, weights=Pk_field)[0]
    Nmodes = np.histogram(dist_2d, bins=k_bins)[0]
    Pk     = (Pk/Nmodes)*(SubBoxSize/nc**2)**3
    
    k = k[1:];  Pk = Pk[1:]
    
    return k, Pk



##https://github.com/EmilienDupont/wgan-gp/blob/master/training.py

def _gradient_penalty(D, real_data, generated_data, gp_weight):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return gp_weight * ((gradients_norm - 1) ** 2).mean()
    
def _gradient_penalty_2d(D,real_data, generated_data, gp_weight):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return gp_weight * ((gradients_norm - 1) ** 2).mean()


class HydrogenDataset(Dataset):
    """Hydrogen Dataset"""

    def __init__(self, part, datapath, s_sample,  transform, d2):
        """
        Args:
            h5_file (string): name of the h5 file with 32 sampled cubes.
            root_dir (string): Directory with the .h5 file.
        """
        self.part = part
        self.s_sample = s_sample
        #self.t_coords = {'x': [0, 1023], 'y': [0, 1023], 'z': [0, 1023]} # Hardcoded, can use define_test()
        self.transform=transform
        self.datapath=datapath
        self.d2=d2

    def __len__(self):
        # Function called when len(self) is executed
        return len(self.part['train'])

    def __getitem__(self, index):

        idx = self.part['train'][index]
        
        sample = torch.load(self.datapath + "/sample_" + str(idx) + ".pickle")
        sample = np.array(sample)
        
        if self.transform!=None:
            sample = data_transform(sample, self.transform, inverse=False)
                  
                
        if self.d2 == False:
            sample = sample.reshape((1, self.s_sample, self.s_sample, self.s_sample))
        else:
            sample = sample.mean(axis=2)  
            sample = sample.reshape((1, self.s_sample, self.s_sample))

        return torch.tensor(sample)


def get_samples(file, s_sample, nsamples, test_coords):
    #n is size of minibatch, get valid samples (not intersecting with test_coords)
    sample_list=[]
    m=2048-s_sample
    for n in range(nsamples):
        #print("Sample No = " + str(n + 1) + " / " + str(nsamples))
        sample_valid=False
        while sample_valid==False:
            x = random.randint(0,m)
            y = random.randint(0,m)
            z = random.randint(0,m)
            sample_coords = {'x':[x,x+s_sample], 
                             'y':[y,y+s_sample], 
                             'z':[z,z+s_sample]}
            
            sample_valid = check_coords(test_coords, sample_coords)
        
        sample_list.append(sample_coords)

    sample_array=[]
    #f file has to be opened outisde the function
    for c in sample_list:
        a = file[c['x'][0]:c['x'][1],
              c['y'][0]:c['y'][1],
              c['z'][0]:c['z'][1]]
        
        sample_array.append(np.array(a))
    
    return np.array(sample_array)


def data_transform(sample, transform, inverse=False):
    e = 1e-2
    em = 1
    
    if transform == 'log_max':
        if inverse == False:
            sample = np.log10(sample + e) / (np.log10(max_5) - em)
        else:
            sample =10**(sample * (np.log10(max_5) - em)) - e
            
    if transform == 'log_max_p':
        if inverse == False:
            sample = np.log(sample +1) / (max_l5)
        else:
            sample =np.exp(sample * max_l5) - 1
        
    
    return sample




