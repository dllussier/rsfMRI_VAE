#!/usr/bin/env python3

import dgl
import networkx as nx
import numpy as np
import matplotlib.pylab as plt

from nilearn import datasets
from nilearn import plotting
from nilearn import input_data
from nilearn.connectome import ConnectivityMeasure

def plot_matrices(matrices, matrix_kind):
    n_matrices = len(matrices)
    fig = plt.figure(figsize=(n_matrices * 4, 4))
    for n_subject, matrix in enumerate(matrices):
        plt.subplot(1, n_matrices, n_subject + 1)
        matrix = matrix.copy()  # avoid side effects
        # Set diagonal to zero, for better visualization
        np.fill_diagonal(matrix, 0)
        vmax = np.max(np.abs(matrix))
        title = '{0}, subject {1}'.format(matrix_kind, n_subject)
        plotting.plot_matrix(matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
                             title=title, figure=fig, colorbar=False)

#fetch dataset
adhd_data = datasets.fetch_adhd(n_subjects=20)

#input default mode network coordinates (mni)
dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
labels = [
          'Posterior Cingulate Cortex',
          'Left Temporoparietal Junction',
          'Right Temporoparietal Junction',
          'Medial Prefrontal Cortex',
         ]

#extract signal from sphere surrounding dmn cooridinates
masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=8, t_r=2.5, detrend=True, standardize=True,
    low_pass=.1, high_pass=.01, memory='nilearn_cache', memory_level=1)


#compute signal and extract phenotype information
adhd_subjects = []
pooled_subjects = []
site_names = []
adhd_labels = []  # 1 if ADHD, 0 if control
for func_file, confound_file, phenotypic in zip(
        adhd_data.func, adhd_data.confounds, adhd_data.phenotypic):
    time_series = masker.fit_transform(func_file, confounds=confound_file)
    pooled_subjects.append(time_series)
    is_adhd = phenotypic['adhd']
    if is_adhd:
        adhd_subjects.append(time_series)

    site_names.append(phenotypic['site'])
    adhd_labels.append(is_adhd)

print('Data has {0} ADHD subjects.'.format(len(adhd_subjects)))

#create correlation matrices
correlation_measure = ConnectivityMeasure(kind='correlation')

correlation_matrices = correlation_measure.fit_transform(adhd_subjects)

#view 2d matrix shape
print('Correlations of ADHD patients are stacked in an array of shape {0}'
      .format(correlation_matrices.shape))

#show connectivity matrices of first 3 subjects
plot_matrices(correlation_matrices[:3], 'correlation')


#draw connectome to networkx graph
G = nx.MultiGraph()
a = correlation_matrices[3]
D = nx.to_networkx_graph(a, create_using=nx.MultiGraph)
keys = G.add_edges_from(D.edges)

#view networkx graph
nx.draw(G, with_labels=True)

#convert nx graph to dgl
g_dgl = dgl.DGLGraph(G)

#view dgl graph
nx.draw(g_dgl.to_networkx(), with_labels=True)
