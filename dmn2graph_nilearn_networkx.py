#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn import input_data
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure


#import dataset
abide_dataset = datasets.fetch_abide_pcp(derivatives=['func_preproc'],
                        n_subjects=1)

func_filename = abide_dataset.func_preproc[0]

#print basic information on the dataset
print('First functional nifti image (4D) is at: %s' % #location of image
      func_filename[0])  
print(abide_dataset.keys())

#input default mode network coordinates (mni)
dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
labels = [
          'Posterior Cingulate Cortex',
          'Left Temporoparietal Junction',
          'Right Temporoparietal Junction',
          'Medial Prefrontal Cortex',
         ]

#extracts signal from sphere surrounding dmn cooridinates
masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=8,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2.5,
    memory='nilearn_cache', memory_level=1, verbose=2)

time_series = masker.fit_transform(func_filename)

#dsplay time series
for time_serie, label in zip(time_series.T, labels):
    plt.plot(time_serie, label=label)

plt.title('Default Mode Network Time Series')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.legend()
plt.tight_layout()

#compute partial correlation matrix
connectivity_measure = ConnectivityMeasure(kind='partial correlation')
partial_correlation_matrix = connectivity_measure.fit_transform(
    [time_series])[0]

#display connectome
plotting.plot_connectome(partial_correlation_matrix, dmn_coords,
                         title="Default Mode Network Connectivity")


#display connectome with hemispheric projections
plotting.plot_connectome(partial_correlation_matrix, dmn_coords,
                         title="Connectivity projected on hemispheres",
                         display_mode='lyrz')
plotting.show()

#draw connectome to networkx graph
G = nx.MultiGraph()
G.add_nodes_from(dmn_coords)
a = partial_correlation_matrix
D = nx.to_networkx_graph(a, create_using=nx.MultiGraph)
keys = G.add_edges_from(D.edges)
