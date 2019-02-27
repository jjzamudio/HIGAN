from torch.utils.data import Dataset, DataLoader
import os
import random
import numpy as np
from utils.data_utils import *
import torch

def define_test(s_test, s_train):
    """
    s_test = one edge of the test partition of the whole cube
    s_train = one edge of the sampled subcubes
    """
    #2048/16=128
    m = 8
    x = random.randint(0,m) * s_train
    y = random.randint(0,m) * s_train
    z = random.randint(0,m) * s_train
    #print(x,y,z)
    return {'x':[x,x + s_test], 'y':[y,y + s_test], 'z':[z,z + s_test]}

def check_coords(test_coords, train_coords):
    valid=True
    for i in ['x','y','z']:
        r=(max(test_coords[i][0], 
               train_coords[i][0]), 
           min(test_coords[i][1],
               train_coords[i][1]))
        if r[0]<=r[1]:
            valid=False
    return valid

def rotations24(polycube):
    # imagine shape is pointing in axis 0 (up)

    # 4 rotations about axis 0
    yield from rotations4(polycube, 0)

    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    yield from rotations4(rot90(polycube, 2, axis=1), 0)

    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    yield from rotations4(rot90(polycube, axis=1), 2)
    yield from rotations4(rot90(polycube, -1, axis=1), 2)

    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    yield from rotations4(rot90(polycube, axis=2), 1)
    yield from rotations4(rot90(polycube, -1, axis=2), 1)

def rotations4(polycube, axis):
    """List the four rotations of the given cube about the given axis."""
    for i in range(4):
        yield rot90(polycube, i, axis)
        
def rot90(m, k=1, axis=2):
    """Rotate an array k*90 degrees in the counter-clockwise direction around the given axis"""
    m = numpy.swapaxes(m, 2, axis)
    m = numpy.rot90(m, k)
    m = numpy.swapaxes(m, 2, axis)
    return m


def get_samples(s_sample, 
                nsamples, 
#                 h5_filename,  
                test_coords,
                f):   # given as f["delta_HI"]
    #n is size of minibatch, get valid samples (not intersecting with test_coords)
#     sample_list=[]
    m = 2048 - s_sample
    
    
    for n in range(nsamples):
        #print("Sample No = " + str(n + 1) + " / " + str(nsamples))
        sample_valid = False
        while sample_valid == False:
            x = random.randint(0,m)
            y = random.randint(0,m)
            z = random.randint(0,m)
            sample_coords = {'x':[x,x+s_sample], 
                             'y':[y,y+s_sample], 
                             'z':[z,z+s_sample]}
            
            sample_valid = check_coords(test_coords, 
                                        sample_coords)
        
#         sample_list.append(sample_coords)
    
#     print("Sampling subcube edges finished.")
        
    #Load cube and get samples and convert them to np.arrays
#     sample_array=[]
    

#     time_1 = time.time()
#     f = h5py.File(h5_filename, 'r') 
#     f=f_deltaHI
    
#     counter = 0
#     for c in sample_list:
#     c = sample_list[0]
    c = sample_coords
#         print("Counter = " + str(counter + 1) + " / " + str(len(sample_list)))
    a = f[c['x'][0]:c['x'][1],
          c['y'][0]:c['y'][1],
          c['z'][0]:c['z'][1]]
#         time_2 = time.time()
        
#     a = np.array(a)
        
    
#     sample_array.append(a)
    
#         counter = counter + 1
    
#     time_3 = time.time() - time_2
#     time_2 = time_2 - time_1
    
#     print("time_2 = " + str(time_2))
#     print("time_3 = " + str(time_3))
#     time_2_list.append(time_2)
#     time_3_list.append(time_3)
#     print_count = random.randint(a=0,b=40)
#     if print_count % 40 == 0:
#         print("time_2 mean = " + str(np.mean(np.array(time_2_list))))
#         print("time_3 mean = " + str(np.mean(np.array(time_3_list))))

        
#     f = 0
#     return sample_array
    return a




class HydrogenDataset(Dataset):
    """Hydrogen Dataset"""

    def __init__(self, h5_file, f, root_dir, s_test, s_train,
                 s_sample, nsamples, min_cube, max_cube, mean_cube, stddev_cube,
                min_raw_cube,max_raw_cube,mean_raw_cube,stddev_raw_cube, rotate_cubes, transform, root, dimensions):
        """
        Args:
            h5_file (string): name of the h5 file with 32 sampled cubes.
            root_dir (string): Directory with the .h5 file.
        """
        file_size = os.path.getsize(root_dir + h5_file) / 1e6 # in MBs
#         print("The whole file size is " + str(int(file_size)) + " MBs")
        
        # self.subcubes = h5py.File('../data/sample_32.h5', 'r')
#         self.f = f_deltaHI
        self.h5_file = h5_file
        self.root_dir = root_dir
#         self.f = f
        self.f_open = False
        self.s_test = s_test
        self.s_train = s_train
        self.t_coords = define_test(self.s_test,
                                    self.s_train)
        self.s_sample = s_sample
        self.nsamples = nsamples
        self.h5_filename = self.root_dir + self.h5_file
        
#         self.samples = get_samples(s_sample = self.s_sample,
#                              nsamples = self.nsamples,
#                              h5_filename = self.h5_filename,
#                              test_coords = self.t_coords)
#         print("Got self.samples")
        
        # Transformed Summary Statistics
        self.min_val = min_cube
#         print("min = " + str(self.min_val))
        self.max_val = max_cube
#         print("max = " + str(self.max_val))
        self.mean_val = mean_cube
        self.stddev_val = stddev_cube
        
        # Raw Data Summary Statistics
        self.min_raw_val = min_raw_cube
        self.max_raw_val = max_raw_cube
        self.mean_raw_val = mean_raw_cube
        self.stddev_raw_val = stddev_raw_cube
        
        # Whether to rotate the sampled subcubes
        self.rotate_cubes = rotate_cubes
        
        # transform
        self.transform = transform
        self.root = root
        
        # 2D or 3D
        self.dimensions = dimensions
        

    def __len__(self):
        # Function called when len(self) is executed
        
        #print(len(self.subcubes))
#         return len(self.nsamples)
        return self.nsamples

    def __getitem__(self, idx):
        
        random_sample = np.random.randint(low = 0, high = 20000)
        
        sample = torch.load("/samples/sample_" + str(random_sample) + ".pickle")
        sample = np.array(sample)

        #transforms happens here
        # scale_01 / scale_neg11 / root / root_scale_01 / root_scale_neg11
        sample = transform_func(cube = sample,
                                inverse_type = self.transform,
                                self = self)
        
        if self.dimensions == "3D":
            sample = np.array(sample).reshape((1,
                                               self.s_sample,
                                               self.s_sample,
                                               self.s_sample))
        elif self.dimensions == "2D":
            # make 3D cube into 2D
            sample = sample.mean(axis = 1)

            sample = np.array(sample).reshape((1,
                                               self.s_sample,
                                               self.s_sample))
        
        sample = torch.from_numpy(sample)

        return sample