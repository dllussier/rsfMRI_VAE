#!/usr/bin/env python3
import torch
import os
import re
import shutil
import dgl
import networkx as nx
import matplotlib.pylab as plt
import numpy as np
from nilearn import datasets
from nilearn import plotting 
from glob import glob
from tqdm import tqdm
from shutil import copyfile
from sklearn.model_selection import train_test_split
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
correlation_measure = ConnectivityMeasure(kind='correlation')

for s in [train_dir,test_dir]:
    func_files = glob(os.path.join(s,"*_func_preproc.nii.gz"))    
    for idx in tqdm(range(len(func_files))):
        func_data = func_files[idx]
        sub_name = re.findall(r'_\d+',func_data)[0]
        
        #extract time series
        time_series = masker.fit_transform(func_data, confounds=None)
        print(time_series.shape)
        
        #create correlation matrices and view shape
        correlation_matrix = correlation_measure.fit_transform([time_series])
        print('Correlations are in an array of shape {0}'.format(correlation_matrix.shape))
        print(correlation_matrix)

        #save correlation matrix as numpy file
        corr_save = os.path.join(s, f'{sub_name}_correlations')
        np.save(corr_save, correlation_matrix, allow_pickle=True, fix_imports=True)

        #show connectivity matrix plot
        plot_matrices(correlation_matrix, 'correlation')
        # Mask out the major diagonal
        #np.fill_diagonal(correlation_matrix, 0)
        #plotting.plot_matrix(correlation_matrix, labels=labels, colorbar=True,vmax=0.8, vmin=-0.8)
        
    #remove original 4D files
    for f in func_files:      
        os.remove(f)


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
