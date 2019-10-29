#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lussier

uses previously save .npy files to generate graphs that are saved for use by dataloader
"""

import os
import re
import shutil
import dgl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting 
from glob import glob
from tqdm import tqdm
from shutil import copyfile
from sklearn.model_selection import train_test_split
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure


#draw connectome to networkx graph
G = nx.MultiGraph()

#load numpy array from saved .npy file
a = np.load('./data/train/_0050964_correlations.npy', allow_pickle=False)
a
a.shape

#reshape stacked numpy array to 2d 
b = np.reshape(a, (39,39), order='C')
b
b.shape

#verify that the matrices have not changed
plot_matrices(a, 'original')
plot_matrices(a, 'reshaped')

#convert reshaped numpy array to networkx graph 
D = nx.nx.convert.to_networkx_graph(b, create_using=nx.MultiGraph)
keys = G.add_edges_from(D.edges)

#view networkx graph
nx.draw(G, with_labels=True)

#verify number of nodes is consistent with numpy shape
nx.number_of_nodes(G)



#convert nx graph to dgl
g_dgl = dgl.DGLGraph(G)

#view dgl graph
nx.draw(g_dgl.to_networkx(), with_labels=True)
