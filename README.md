# rsfMRI_VAE

Goals:
Build 2D, 3D and graph convolutional VAE frameworks for use with resting-state fMRI data.

VAEs are based off of the Pytorch VAE example: https://github.com/pytorch/examples/tree/master/vae

Requirements:
torch==1.0.1
torchmed==0.0.1a0
sklearn==0.22.1
nibabel==3.0.0
numpy==1.18.1
nilearn==0.6.1

For graph frameworks (in addition to above):
dgl==0.4.2
networkx==2.4
