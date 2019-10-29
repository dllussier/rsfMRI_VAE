#!/usr/bin/env python3
"""
@author: lussier

uses previously save .npy files to generate graphs that are saved for use by dataloader
"""

import os
import re
import shutil
import dgl
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting 
from glob import glob
from tqdm import tqdm
from shutil import copyfile

#import atlas
atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']
# Loading atlas data stored in 'labels'
labels = atlas['labels']

#set up matrix plotting
def plot_matrices(matrices, matrix_kind):
    n_matrices = len(matrices)
    fig = plt.figure(figsize=(n_matrices * 4, 4))
    for n_subject, matrix in enumerate(matrices):
        plt.subplot(1, n_matrices, n_subject + 1)
        matrix = matrix.copy()  # avoid side effects
        #np.fill_diagonal(matrix, 0)
        vmax = np.max(np.abs(matrix))
        title = '{0}, subject {1}'.format(matrix_kind, n_subject)
        plotting.plot_matrix(matrix, labels=labels, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
                             title=title, figure=fig, colorbar=False)

train_dir = './data/train/'
test_dir = './data/test/'

for s in [train_dir,test_dir]:
    array_files = glob(os.path.join(s,"*_correlations.npy"))    
    for idx in tqdm(range(len(array_files))):
        array_data = array_files[idx]
        array_name = re.findall(r'_\d+',array_data)[0]

        #draw connectome to networkx graph
        G = nx.MultiGraph()

        #load numpy array from saved .npy file
        a = np.load(array_data, allow_pickle=True)
        print('Saved correlations are in an array of shape {0}'.format(a.shape))
        print(a)

        #reshape stacked numpy array to 2d 
        b = np.reshape(a, (39,39), order='C')
        print('Reshaped correlation matrix is in an array of shape {0}'.format(b.shape))
        print(b)

        #verify that the matrices have not changed
        #plot_matrices(a, 'original')
        #plot_matrices(b, 'reshaped')
        
        #convert reshaped numpy array to networkx graph 
        D = nx.nx.convert.to_networkx_graph(b, create_using=nx.MultiGraph)
        keys = G.add_edges_from(D.edges)
        
        #verify number of nodes is consistent with numpy shape
        print('For the graph converted from %s the node count is {0}'.format(nx.number_of_nodes(G)) % array_data)
        
        #view networkx graph
        plt.figure()
        nx.draw(G, with_labels=True)
        #plt.show()
        
        #save graph as file for use by dataloader
        array_save = os.path.join(s, f'{array_name}.gpickle')
        nx.write_gpickle(G, array_save)   
        
        #convert nx graph to dgl
        g_dgl = dgl.DGLGraph(G)
        
        #view dgl graph
        plt.figure()
        nx.draw(g_dgl.to_networkx(), with_labels=True)
        #plt.show()
