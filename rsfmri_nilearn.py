#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:02:33 2019

@author: lussier
"""
import numpy as np
from nilearn import datasets
from nilearn import image 
from nilearn import plotting 
from nilearn import decomposition
from nilearn.regions import RegionExtractor
from nilearn.connectome import ConnectivityMeasure


#import dataset
abide = datasets.fetch_abide_pcp(derivatives=['func_preproc','rois_cc200'],
                        SITE_ID=['NYU'],
                        n_subjects=10)

func_filenames = abide.func_preproc #4D data

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' % #location of image
      func_filenames[0])  
print(abide.keys())

#canica decomposition for sample
canica = decomposition.CanICA(n_components=20, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=2,
                threshold=3., verbose=10, random_state=0,
                mask_strategy='background')
canica.fit(func_filenames)

#retrieve components and project back into 3D space then save as nifti
components = canica.components_
components_img = canica.masker_.inverse_transform(components)
components_img.to_filename('canica_resting_state.nii.gz')

#visualize components on map
plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                         title='CanICA components')

#plot generated atlas for single component
plotting.plot_stat_map(image.index_img(components_img, 9), title='9')
plotting.show()

#plot maps for ica components separately
for i, cur_img in enumerate(image.iter_img(components_img)):
    plotting.plot_stat_map(cur_img, display_mode="z", title="IC %d" % i,
                  cut_coords=1, colorbar=False)

plotting.show()

#region extraction from component map
extractor = RegionExtractor(components_img, threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True, min_region_size=1350)
extractor.fit()

regions_extracted_img = extractor.regions_img_ # extracted regions
regions_index = extractor.index_ #region index
n_regions_extracted = regions_extracted_img.shape[-1] #total regions extracted

#visualize extracted regions
title = ('%d regions are extracted from %d components.'
         '\nEach separate color of region indicates extracted region'
         % (n_regions_extracted, 20))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title)

#validate results by comparing original and network region side by side
img = image.index_img(components_img, 4)
coords = plotting.find_xyz_cut_coords(img)
display = plotting.plot_stat_map(img, cut_coords=coords, colorbar=False,
                                 title='Single network')

regions_indices_of_map3 = np.where(np.array(regions_index) == 4)

display = plotting.plot_anat(cut_coords=coords,
                             title='Network regions')

colors = 'rgbcmyk'
for each_index_of_map3, color in zip(regions_indices_of_map3[0], colors):
    display.add_overlay(image.index_img(regions_extracted_img, each_index_of_map3),
                        cmap=plotting.cm.alpha_cmap(color))

plotting.show()

#compute functional connectivity matrices
correlations = []
connectome_measure = ConnectivityMeasure(kind='correlation')
for filename in func_filenames:
    timeseries_each_subject = extractor.transform(filename)
    correlation = connectome_measure.fit_transform([timeseries_each_subject])
    correlations.append(correlation)

mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted,
                                                          n_regions_extracted)

#visualization of matrices
title = 'Correlations between %d regions' % n_regions_extracted
display = plotting.plot_matrix(mean_correlations, vmax=1, vmin=-1,
                               colorbar=True, title=title)

#find the center of the regions and plot the connectome
regions_img = regions_extracted_img
coords_connectome = plotting.find_probabilistic_atlas_cut_coords(regions_img)

plotting.plot_connectome(mean_correlations, coords_connectome,
                         edge_threshold='90%', title=title)



