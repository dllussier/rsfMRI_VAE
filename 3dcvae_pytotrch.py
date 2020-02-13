#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: desiree lussier

3D convolutional varational autoencoder.
takes 3D nifti images as input.
"""

import os
import numpy as np
import torch
import torchmed
import torch.utils.data
from torch import nn, optim
torch.set_default_tensor_type(torch.cuda.DoubleTensor)  #comment out for cpu
from glob import glob
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

#set parameters here for convenience
CUDA=True           #for cpu set to 'False' and for gpu set 'True'
SEED=1
BATCH_SIZE=1
LOG_INTERVAL=10
EPOCHS=1
ZDIMS=500            #bottleneck connections
CLASSES=13
OPT_LEARN_RATE=0.001
STEP_SIZE=1 
GAMMA=0.9 
HDIM=1152
IMG=1               #set to 1 for grayscale and 3 for rgb

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

train_dir = "../data/train/"
test_dir = "../data/test/"
logfile = './cvae_log.txt'

#create customized dataset
class CustomDataset(Dataset):    
    def __init__(self,data_root):
        self.samples = []

        for name in glob(os.path.join(data_root,'*.nii.gz')):
            self.samples.append(name) 

        print('data root: %s' % data_root)
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx): 
        name = self.samples[idx]
        print('name is %s' % name)
        image = torchmed.readers.SitkReader(name)       #read nifti using torchmed
        npimage = image.to_numpy()                      #convert to numpy array
        expanded = np.expand_dims(npimage, axis=0)      #add image dimension
        img = torch.from_numpy(expanded)                #transform image to tensor
        return img 

#define flatten and unflatten
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=HDIM):
        return input.view(input.size(0), size, 1, 1, 1)

#build variational autoencoder model
class VAE(nn.Module):
    def __init__(self, image_channels=IMG, h_dim=HDIM, z_dim=ZDIMS, n_classes=CLASSES):
        super(VAE, self).__init__()
        
        channels = (16,32,96)
                
        print("VAE")
        #encoder layers
        self.conv1 = nn.Conv3d(image_channels, channels[0], kernel_size=2)
        self.conv2 = nn.Conv3d(channels[0], channels[1], kernel_size=2)
        self.conv3 = nn.Conv3d(channels[1], channels[2], kernel_size=2)
        self.conv4 = nn.Conv3d(channels[2], channels[2], kernel_size=2)

        self.maxpool = nn.MaxPool3d(kernel_size=2, return_indices=True) #pooling layers return indices
        
        self.flatten = Flatten()                                        #flattens dims into tensor
        
        self.mu = nn.Linear(h_dim, z_dim)                               #mu layer
        self.logvar = nn.Linear(h_dim, z_dim)                           #logvariance layer

        #decoder layers        
        self.linear = nn.Linear(z_dim, h_dim)                           #pulls from bottleneck to hidden
        self.unflatten = UnFlatten()                                    #unflattens tensor to dims

        self.maxunpool = nn.MaxUnpool3d(kernel_size=2)                  #unpooling layers require indices from pooling layers
        
        self.conv_tran4 = nn.ConvTranspose3d(h_dim, channels[2], kernel_size=(2,3,2))
        self.conv_tran3 = nn.ConvTranspose3d(channels[2], channels[2], kernel_size=(2,2,2))
        self.conv_tran2 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=(3,2,3))
        self.conv_tran1 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=(2,2,3))
        self.conv_tran0 = nn.ConvTranspose3d(channels[0], image_channels, kernel_size=(3,3,2))

        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x):
        h = F.relu(self.conv1(x))
        h, indices1 = self.maxpool(h)
        h = F.relu(self.conv2(h))
        h, indices2 = self.maxpool(h)
        h = F.relu(self.conv3(h))
        h, indices3 = self.maxpool(h)
        h = F.relu(self.conv4(h))
        h, indices4 = self.maxpool(h)
        h = self.flatten(h)
        mu, logvar = self.mu(h), self.logvar(h)
        std = logvar.mul(0.5).exp_()        #reparametization
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z, mu, logvar, indices1, indices2, indices3, indices4   

    def decode(self, z, indices1, indices2, indices3, indices4):
        z = self.linear(z)
        z = self.unflatten(z)
        z = F.relu(self.conv_tran4(z)) 
        z = self.maxunpool(z, indices4)
        z = F.relu(self.conv_tran3(z))
        z = self.maxunpool(z, indices3)
        z = F.relu(self.conv_tran2(z))
        z = self.maxunpool(z, indices2)
        z = F.relu(self.conv_tran1(z))
        z = self.maxunpool(z, indices1)
        z = F.relu(self.conv_tran0(z))
        z = self.sigmoid(z)
        return z
    
    def forward(self, x):
        z, mu, logvar, indices1, indices2, indices3, indices4 = self.encode(x)
        z = self.decode(z, indices1, indices2, indices3, indices4)
        return z, mu, logvar, indices1, indices2, indices3, indices4

model = VAE()
if CUDA:
    model.cuda()
  
#load previous state 
model.load_state_dict(torch.load('3dcvae.torch', map_location='cpu'))

#loss function is reconstruction + KL divergence losses summed over all elements and batch
# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014; https://arxiv.org/abs/1312.6114
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

#optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=OPT_LEARN_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = STEP_SIZE, gamma = GAMMA)

#load dataset using custom dataset and parameters from above
trainset = CustomDataset(train_dir)
testset = CustomDataset(test_dir)

train_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True,  **kwargs)
test_loader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False,  **kwargs)

#train and test model
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,_) in enumerate(train_loader):
        data = Variable(data)
        if CUDA:
            data = data.cuda()
        scheduler.step()
        optimizer.zero_grad()
        recon_batch, mu, logvar, indices1, indices2, indices3, indices4 = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data)))

    #print loss
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data,_) in enumerate(test_loader):
        if CUDA:
            data = data.cuda()
        data = Variable(data, requires_grad=False)
        recon_batch, mu, logvar, indices1, indices2, indices3, indices4 = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data
        
        #print footer and header of target and recontructed voxel values to log file for numerical comparison
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n],recon_batch.view(BATCH_SIZE, IMG, 52, 64, 53)[:n]])         
          comparison = comparison.data.cpu()
          comparison = comparison.detach().numpy()
          comparison = np.squeeze(comparison)
          with open(logfile, 'a') as log:
              print(comparison.shape, file=log)
              print(comparison.size, file=log)
              print(comparison, file=log)
    
    #print loss            
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
#run train and test
if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)

#save model state
torch.save(model.state_dict(), '3dcvae.torch')
