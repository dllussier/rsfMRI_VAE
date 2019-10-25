#!/usr/bin/env python3

import torch
import os
import re
import shutil
import dgl
import networkx as nx
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import torch.utils.data
from nilearn import datasets
from nilearn import image 
from nilearn import plotting 
from nilearn import decomposition
from glob import glob
from tqdm import tqdm
from shutil import copyfile
from nibabel import load as load_fmri
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from nilearn.image import resample_img
from nilearn.input_data import NiftiMasker
from nilearn.input_data import NiftiMapsMasker
from nilearn.regions import RegionExtractor
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import GraphLassoCV


def plot_matrices(matrices, matrix_kind):
    n_matrices = len(matrices)
    fig = plt.figure(figsize=(n_matrices * 4, 4))
    for n_subject, matrix in enumerate(matrices):
        plt.subplot(1, n_matrices, n_subject + 1)
        matrix = matrix.copy()  # avoid side effects
        np.fill_diagonal(matrix, 0)
        vmax = np.max(np.abs(matrix))
        title = '{0}, subject {1}'.format(matrix_kind, n_subject)
        plotting.plot_matrix(matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
                             title=title, figure=fig, colorbar=False)


#import atlas
atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']
# Loading atlas data stored in 'labels'
labels = atlas['labels']

#import dataset
data = datasets.fetch_abide_pcp(derivatives=['func_preproc'], n_subjects=5)

func = data.func_preproc #4D data

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' % #location of image
      func[0])  
print(data.keys())

#create needed directories
func_dir = './data/'
train_dir = './data/train/'
test_dir = './data/test/'
for p in [func_dir,train_dir,test_dir]:
    if not os.path.exists(p):
        os.mkdir(p)
#move functional data to local data directory
for f in func:
    shutil.move(f, func_dir)

#randomize and split training and test data 
all_files = glob(os.path.join(func_dir,"*.nii.gz"))

train,test = train_test_split(all_files,test_size = 0.2,random_state = 12345, shuffle=True)

for t in tqdm(train):
    copyfile(t,os.path.join(train_dir,os.path.split(t)[1]))
    
for t in tqdm(test):
    copyfile(t,os.path.join(test_dir,os.path.split(t)[1]))

#designate mask
masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                         memory='nilearn_cache', verbose=5)


for s in [train_dir,test_dir]:
    graphs_dir  = os.path.join(s,'graphs/')
    func_files = glob(os.path.join(s,"*_func_preproc.nii.gz"))
    if not os.path.exists(graphs_dir):
        os.mkdir(graphs_dir)
    print(graphs_dir)
    
    for idx in tqdm(range(len(func_files))):
        func_data = func_files[idx]
        sub_name = re.findall(r'_\d+',func_data)[0]
        
        #extract timr series
        time_series = masker.fit_transform(func_data, confounds=None)

        #create correlation matrices
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform(func_data)

        #view matrices array shape and data
        print('Correlations are in an array of shape {0}'.format(correlation_matrix.shape))
        print(correlation_matrix)
        #save correlation matrix as 
        numpy.save(file, correlation_matrix, allow_pickle=True, fix_imports=True)
        #show connectivity matrix plot
        plot_matrices(correlation_matrix, 'correlation')


#draw connectome to networkx graph
G = nx.MultiGraph()
a = correlation_matrix
D = nx.to_networkx_graph(a, create_using=nx.MultiGraph)
keys = G.add_edges_from(D.edges)

#view networkx graph
nx.draw(G, with_labels=True)

#convert nx graph to dgl
g_dgl = dgl.DGLGraph(G)

#view dgl graph
nx.draw(g_dgl.to_networkx(), with_labels=True)
