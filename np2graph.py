#!/usr/bin/env python3
"""
@author: lussier

uses previously save .npy files to generate graphs that are saved for use by dataloader
"""

import os
import re
import dgl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting 
from glob import glob
from tqdm import tqdm

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
#define site folders
train_dir = './data/train/'
test_dir = './data/test/'
pitt_dir = './data/train/pitt/'
olin_dir = './data/train/olin/'
ohsu_dir = './data/train/ohsu/'
sdsu_dir = './data/train/sdsu/'
trinity_dir = './data/train/trinity/'
um_1_dir = './data/train/um_1/'
um_2_dir = './data/train/um_2/'
usm_dir = './data/train/usm/'
yale_dir = './data/train/yale/'
cmu_dir = './data/train/cmu/'
leuven_1_dir = './data/train/leuven_1/'
leuven_2_dir = './data/train/leuven_2/'
kki_dir = './data/train/kki/'
nyu_dir = './data/train/nyu/'
stanford_dir = './data/train/stanford'
ucla_1_dir = './data/train/ucla_1/'
ucla_2_dir = './data/train/ucla_2/'
maxmun_dir = './data/train/maxmun/'
caltech_dir = './data/train/caltech/'
sbl_dir = './data/train/sbl/'
pitt_test_dir = './data/test/pitt/'
olin_test_dir = './data/test/olin/'
ohsu_test_dir = './data/test/ohsu/'
sdsu_test_dir = './data/test/sdsu/'
trinity_test_dir = './data/test/trinity/'
um_1_test_dir = './data/test/um_1/'
um_2_test_dir = './data/test/um_2/'
usm_test_dir = './data/test/usm/'
yale_test_dir = './data/test/yale/'
cmu_test_dir = './data/test/cmu/'
leuven_1_test_dir = './data/test/leuven_1/'
leuven_2_test_dir = './data/test/leuven_2/'
kki_test_dir = './data/test/kki/'
nyu_test_dir = './data/test/nyu/'
stanford_test_dir = './data/test/stanford'
ucla_1_test_dir = './data/test/ucla_1/'
ucla_2_test_dir = './data/test/ucla_2/'
maxmun_test_dir = './data/test/maxmun/'
caltech_test_dir = './data/test/caltech/'
sbl_test_dir = './data/test/sbl/'

for s in [pitt_dir,olin_dir,ohsu_dir,sdsu_dir,trinity_dir,um_1_dir,um_2_dir,
          usm_dir,yale_dir,cmu_dir,leuven_1_dir,leuven_2_dir,kki_dir,nyu_dir,
          stanford_dir,ucla_1_dir,ucla_2_dir,maxmun_dir,caltech_dir,sbl_dir,
          pitt_test_dir,olin_test_dir,ohsu_test_dir,sdsu_test_dir,
          trinity_test_dir,um_1_test_dir,um_2_test_dir,usm_test_dir,
          yale_test_dir,cmu_test_dir,leuven_1_test_dir,leuven_2_test_dir,
          kki_test_dir,nyu_test_dir,stanford_test_dir,ucla_1_test_dir,
          ucla_2_test_dir,maxmun_test_dir,caltech_test_dir, sbl_test_dir]:
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
        
