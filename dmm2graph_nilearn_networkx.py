#!/usr/bin/env python3
#edited from http://nilearn.github.io/auto_examples/03_connectivity/plot_adhd_spheres.html

from nilearn import datasets
from nilearn import input_data
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting


#Retrieve the dataset
adhd_dataset = datasets.fetch_adhd(n_subjects=1)
#import dataset
abide_dataset = datasets.fetch_abide_pcp(derivatives=['func_preproc'],
                        n_subjects=1)

func_filename = abide_dataset.func_preproc[0]

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' % #location of image
      func_filename[0])  
print(abide_dataset.keys())

#Coordinates of Default Mode Network
dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
labels = [
          'Posterior Cingulate Cortex',
          'Left Temporoparietal junction',
          'Right Temporoparietal junction',
          'Medial prefrontal cortex',
         ]

#Extracts signal from sphere around DMN seeds
masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=8,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2.5,
    memory='nilearn_cache', memory_level=1, verbose=2)

#confound_filename = abide_dataset.confounds[0]

time_series = masker.fit_transform(func_filename)

#Display time series
for time_serie, label in zip(time_series.T, labels):
    plt.plot(time_serie, label=label)

plt.title('Default Mode Network Time Series')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.legend()
plt.tight_layout()


#Compute partial correlation matrix
#Using object nilearn.connectome.ConnectivityMeasure: Its default covariance estimator is Ledoit-Wolf, allowing to obtain accurate partial correlations.
connectivity_measure = ConnectivityMeasure(kind='partial correlation')
partial_correlation_matrix = connectivity_measure.fit_transform(
    [time_series])[0]

#Display connectome
plotting.plot_connectome(partial_correlation_matrix, dmn_coords,
                         title="Default Mode Network Connectivity")


#Display connectome with hemispheric projections. Notice (0, -52, 18) is included in both hemispheres since x == 0.
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
