from __future__ import print_function
import numpy as np
import h5py
import pickle as pkl
import random
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.distributions import normal
import itertools
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import scipy.spatial as sp
import pyfftw
import Pk_library as PKL

NUM_REDSHIFTS = 5

def read_halo_file(catalogue_path = "halo_catalogues/",
                   redshift = 0,
                  ):
    path = "{}Halo_catalogue_z={}.hdf5".format(catalogue_path, 
                                                redshift)
    
    f = h5py.File(path)
    mass = f["mass"]
    position_x = np.array(f["pos"])[:,0]
    position_y = np.array(f["pos"])[:,1]
    position_z = np.array(f["pos"])[:,2]
    radius = f["radius"]
    return np.array(mass), position_x, position_y, position_z, np.array(radius)


def build_redshift_data(halo_catalogue_path):
    redshift_dict = defaultdict(dict)
    for i in range(NUM_REDSHIFTS + 1):
        redshift_dict[i]["total_mass"], redshift_dict[i]["position_x"], redshift_dict[i]["position_y"], \
        redshift_dict[i]["position_z"], redshift_dict[i]["radius"] = \
            [*read_halo_file(halo_catalogue_path, i)] 
        
        redshift_dict[i]["HI_mass"] = np.array(pd.DataFrame(pd.read_csv("MHI_z={}.csv".format(i)))["0"])
        
    [df_0, df_1, df_2, df_3, df_4, df_5] = [pd.DataFrame({"total_mass":redshift_dict[i]["total_mass"],
                                                         "position_x":redshift_dict[i]["position_x"],
                                                         "position_y":redshift_dict[i]["position_y"],
                                                         "position_z":redshift_dict[i]["position_z"],
                                                         "radius":redshift_dict[i]["radius"],
                                                         "HI_mass":redshift_dict[i]["HI_mass"]}) \
                                           for i in range(NUM_REDSHIFTS+1)]
    
    for d in [df_0, df_1, df_2, df_3, df_4, df_5]:
        d["voxel_x"] = d["position_x"].apply(lambda x: np.floor(x*2048/75))
        d["voxel_y"] = d["position_y"].apply(lambda x: np.floor(x*2048/75))
        d["voxel_z"] = d["position_z"].apply(lambda x: np.floor(x*2048/75))
    
    return df_0, df_1, df_2, df_3, df_4, df_5

#######################################
## NON-DENSITY ADJUSTED HOD FOR M_HI ##
#######################################

def get_num_halos(df):
	return len(df)

def get_halo_tensor(pos_x, pos_y, pos_z, redshift, sample_edge):
    
    """Args:
    - pos_x: left corner x coordinate in simulation, (int in (0, 2048-sample_edge))
    - pos_y: left corner y coordinate in simulation, (int in (0, 2048-sample_edge))
    - pos_z: left corner z coordinate in simulation, (int in (0, 2048-sample_edge))
    
    - redshift: redshift to read the correct halo catalogue, (int in range(NUM_REDSHIFTS+1))
    - sample_edge: size of one edge of each sample (int: sample.size[0])"""
    
    new_halo_tensor = np.zeros((sample_edge,)*3)
    redshift_df = pd.DataFrame(pd.read_csv("redshift_df_{}.csv".format(redshift)))

    cube_in_position = redshift_df[(redshift_df["voxel_x"]>=pos_x)&(redshift_df["voxel_x"]<pos_x+sample_edge)&\
                                   (redshift_df["voxel_y"]>=pos_y)&(redshift_df["voxel_y"]<pos_y+sample_edge)&\
                                   (redshift_df["voxel_z"]>=pos_z)&(redshift_df["voxel_z"]<pos_z+sample_edge)]
    
    for row in cube_in_position.index.values:
        vx, vy, vz = int(cube_in_position["voxel_x"].loc[row]), int(cube_in_position["voxel_y"].loc[row]), \
                            int(cube_in_position["voxel_z"].loc[row])
        new_halo_tensor[vx, vy, vz] = new_halo_tensor[vx, vy, vz] + redshift_df["HI_mass"].loc[row]
    path = "large_halo_cubes/redshift_{}".format(redshift)
    edge_r = [*np.arange(0, 2048, 64)]
    lower_corners = [*itertools.product(edge_r, edge_r, edge_r)]
    for corner in lower_corners:
        with open("{}/lower_corner_{}_{}_{}.pkl".format(path, corner[0], corner[1], corner[2]),"wb") as f:
            pkl.dump(new_halo_tensor[corner[0]:corner[0]+64,corner[1]:corner[1]+64,corner[2]:corner[2]+64], 
                     f, protocol = 2)

###################################
## DENSITY-ADJUSTED HOD FOR M_HI ##
###################################

# NEW
def get_particle_position(df, n_particles):
    num_halos = df.shape[0]
    n_particles = int(n_particles)
    center_x, center_y, center_z = df["voxel_x"], df["voxel_y"], df["voxel_z"]
    U = np.random.rand(num_halos, n_particles) 
    V = np.random.rand(num_halos, n_particles) 
    W = np.random.rand(num_halos, n_particles)
    print ("random numbers generated.")
    r = np.multiply(np.array(df["voxel_radius"]).reshape(-1,1), np.power(U, 1/3))
    phi = np.multiply(np.ones((num_halos, 1)), 2*np.pi*V)
    theta = np.multiply(np.ones((num_halos, 1)), np.arccos(-1+2*W))
    print ("r, phi, theta generated.")
    x = np.floor(np.array(center_x).reshape(-1,1) + np.multiply(np.multiply(r, np.sin(theta)), np.cos(phi)))
    y = np.floor(np.array(center_y).reshape(-1,1) + np.multiply(np.multiply(r, np.sin(theta)), np.sin(phi)))
    z = np.floor(np.array(center_z).reshape(-1,1) + np.multiply(r, np.cos(theta)))
    print ("x, y, z generated. stacking started.")
    particles_in_halos = np.stack((x,y,z), axis=2)
    return particles_in_halos

# NEW
def write_mhi_per_particle(df, n_particles):
    df["mhi_per_part"] = df["HI_mass"].apply(lambda x: x/n_particles)
    return df

# NEW
def get_pos_and_particle_list(particles_in_halos, df, n_particles):
    position_density = []
    for i in range(len(df)):
        density = df["mhi_per_part"].iloc[i]
        position_density.extend([density for x in range(n_particles)])
    pos_ = particles_in_halos
    return pos_, position_density

# NEW
SAMPLE_EDGE = 2048
new_halo_tensor = np.zeros((SAMPLE_EDGE,)*3)
def write_sphere_voxels(df, pos_, new_halo_tensor,density_list, sample_edge, subsample_edge):
    for i in range(pos_.shape[0]):
        ix = pos_[i].astype(int)
        if all((ix > 0) & (ix <2048)):
            new_halo_tensor[ix[0], ix[1], ix[2]] = density_list[i]
        if i % 1e6 == 0:
            print ("{}% completed.".format(100*i/pos_.shape[0]))
    path = "halo_density_cubes_5"
    edge_r = [*np.arange(0, sample_edge, subsample_edge)]
    lower_corners = [*itertools.product(edge_r, edge_r, edge_r)]
    for corner in lower_corners:
        with open("{}/lower_corner_{}_{}_{}.pkl".format(path, corner[0], corner[1], corner[2]),"wb") as f:
            pkl.dump(new_halo_tensor[corner[0]:corner[0]+subsample_edge,corner[1]:corner[1]+subsample_edge,corner[2]:corner[2]+subsample_edge],
                     f, protocol = 2)
    return new_halo_tensor

# OLD
def get_pos_and_dens_list(halo_indices_path, df):

	"""Args;
    - halo_indices_path: path to the dumped pkl file that holds tensor indices that are inside halos
    - df: redshift df obtained by build_redshift_data, assumed to be already written in the folder as a csv file
    
    Returns;
    - (pos_)position list, (position_density) corresponding density list
    """
	with open(halo_indices_path,"rb") as f:
	    points_in_halos = pkl.load(f)
	position_density = []
	for i in range(len(df)):
	    density = df["halo_density"].iloc[i]
	    len_halo_points = points_in_halos[i].shape[0]
	    position_density.extend([density for x in range(len_halo_points)])

	pos_ = [*points_in_halos[0]]
	ext = pd.Series(points_in_halos[1:]).apply(lambda x: pos_.extend(x))

	return pos_, position_density

def write_sphere_voxels(df, pos_list, density_list, sample_edge):
    new_halo_tensor = np.zeros((sample_edge,)*3)
    for i in range(len(pos_list)):
        ix = pos_list[i]
        new_halo_tensor[ix[0], ix[1], ix[2]] = density_list[i]
    
    path = "halo_density_cubes_5"
    edge_r = [*np.arange(0, sample_edge, 64)]
    lower_corners = [*itertools.product(edge_r, edge_r, edge_r)]
    for corner in lower_corners:
        with open("{}/lower_corner_{}_{}_{}.pkl".format(path, corner[0], corner[1], corner[2]),"wb") as f:
            pkl.dump(new_halo_tensor[corner[0]:corner[0]+64,corner[1]:corner[1]+64,corner[2]:corner[2]+64], 
                     f, protocol = 2)
    return new_halo_tensor

