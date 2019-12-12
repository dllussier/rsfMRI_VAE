#!/usr/bin/env python3
"""
@author: lussier
uses previously generated .mat files from NIAK connectome pipeline to generate graphs that are saved for use by dataloader
"""

import os
import re
import scipy.io
import networkx as nx
import numpy as np
from glob import glob
from tqdm import tqdm

#define data folders containing connectomes
connectome_dir = './output_abide_connectomes/'
pitt_dir = os.path.join(connectome_dir, 'pitt')
olin_dir = os.path.join(connectome_dir, 'olin')
#ohsu_dir = os.path.join(connectome_dir, 'ohsu')
sdsu_dir = os.path.join(connectome_dir, 'sdsu')
trinity_dir = os.path.join(connectome_dir, 'trinity')
#um_1_dir = os.path.join(connectome_dir, 'um_1')
#um_2_dir = os.path.join(connectome_dir, 'um_2')
usm_dir = os.path.join(connectome_dir, 'usm')
yale_dir = os.path.join(connectome_dir, 'yale')
#cmu_dir = os.path.join(connectome_dir, 'cmu')
leuven_1_dir = os.path.join(connectome_dir, 'leuven_1')
leuven_2_dir = os.path.join(connectome_dir, 'leuven_2')
kki_dir = os.path.join(connectome_dir, 'kki')
nyu_dir = os.path.join(connectome_dir, 'nyu')
#stanford_dir = os.path.join(connectome_dir, 'stanford')
ucla_1_dir = os.path.join(connectome_dir, 'ucla_1')
ucla_2_dir = os.path.join(connectome_dir, 'ucla_2')
maxmun_dir = os.path.join(connectome_dir, 'maxmun')
caltech_dir = os.path.join(connectome_dir, 'caltech')
sbl_dir = os.path.join(connectome_dir, 'sbl')

for s in [pitt_dir,olin_dir,sdsu_dir,trinity_dir,#ohsu_dir,um_1_dir,um_2_dir,cmu_dir,
          yale_dir,usm_dir,leuven_1_dir,leuven_2_dir,kki_dir,nyu_dir,#stanford_dir,
          ucla_1_dir,ucla_2_dir,maxmun_dir,caltech_dir,sbl_dir]:
#    connectome_files = glob(os.path.join(s,"graph_prop/*.mat"))

    graph_prop_dir = os.path.join(s,"graph_prop/")
    graph_mat_file = glob(os.path.join(graph_prop_dir,"*.mat"))
    graph_mat = loadmat(graph_mat_file)
    graph_mat
    
    conn_dir = os.path.join(s,"connectomes/")
    connectome_mat_file = glob(os.path.join(conn_dir,"*.mat"))
    connectome_mat = loadmat(connectome_mat_file)
    connectome_mat    

    for idx in tqdm(range(len(connectome_files))):
        connectomes = connectome_files[idx]
        connectome_name = re.findall(r'graph_prop_rois_sub\d+',connectomes)[0]

        #draw connectome to networkx graph
        G = nx.MultiGraph()

        #load numpy array from saved .npy file
        a = scipy.io.loadmat(connectomes)
        #a = np.load(connectomes, allow_pickle=True)
        print('Saved correlations are in an array of shape {0}'.format(a.shape))
        print(a)

        ###reshape stacked numpy array to 2d 
        #b = np.reshape(a, (39,39), order='C')
        #print('Reshaped correlation matrix is in an array of shape {0}'.format(b.shape))
        #print(b)
        
        #convert reshaped numpy array to networkx graph 
        D = nx.nx.convert.to_networkx_graph(a, create_using=nx.MultiGraph)
        keys = G.add_edges_from(D.edges)
        
        #verify number of nodes is consistent with numpy shape
        print('For the graph converted from %s the node count is {0}'.format(nx.number_of_nodes(G)) % connectomes)
        
        #save graph as file for use by dataloader
        graph_save = os.path.join(s, f'{connectome_name}.gpickle')
        nx.write_gpickle(G, graph_save)   
        print('Graph pickle saved as %s' % graph_save)
