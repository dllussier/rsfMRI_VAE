#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: desiree lussier

"""

import os
import numpy as np
import torch
import torchmed
import torch.utils.data
from torch import nn, optim
torch.set_default_tensor_type(torch.cuda.FloatTensor)  #comment out for cpu
from glob import glob
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from nibabel import load as load_fmri

CUDA=True    #for cpu set to 'False'
SEED=1
BATCH_SIZE=1
LOG_INTERVAL = 10
EPOCHS=50
ZDIMS=24
CLASSES=17 
OPT_LEARN_RATE=0.01
STEP_SIZE=1 
GAMMA=0.9 
HDIM=576
IMG=1       #set to 1 for grayscale and 3 for rgb

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

#load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

train_dir = "./train/"
test_dir = "./test/"

#create customized dataset
class CustomDataset(Dataset):    
    def __init__(self,data_root):
        self.samples = []

        for label in os.listdir(data_root):            
                labels_folder = os.path.join(data_root, label)

                for name in glob(os.path.join(labels_folder,'*.nii.gz')):
                    self.samples.append((label,name)) 

        print('data root: %s' % data_root)
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx): 
        label,name=self.samples[idx]
        print('name is %s' % name)
        image = torchmed.readers.SitkReader(name)
        npimage = image.to_numpy()
        expanded = np.expand_dims(npimage, axis=0)
        img = torch.from_numpy(expanded)
        nplabel=np.asarray(label, dtype='int32')
        return img, nplabel 

#define model
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=HDIM):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=IMG, h_dim=HDIM, z_dim=ZDIMS, n_classes=CLASSES):
        super(VAE, self).__init__()
        
        print("VAE")
        #encoder layers
        self.conv1 = nn.Conv3d(image_channels, 16, kernel_size=2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=2)
        self.conv3 = nn.Conv3d(32, 96, kernel_size=2)
        self.conv4 = nn.Conv3d(96, 96, kernel_size=2)

        self.maxpool = nn.MaxPool3d(kernel_size=2, return_indices=True)
        
        self.flatten = Flatten()
        
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
        
        #decoder layers        
        self.linear = nn.Linear(z_dim, h_dim)  
        self.unflatten = UnFlatten()

        self.maxunpool = nn.MaxUnpool3d(kernel_size=2)
        
        self.conv_tran4 = nn.ConvTranspose3d(h_dim, 96, kernel_size=(2,3))
        self.conv_tran3 = nn.ConvTranspose3d(96, 96, kernel_size=2)
        self.conv_tran2 = nn.ConvTranspose3d(96, 32, kernel_size=(3,2))
        self.conv_tran1 = nn.ConvTranspose3d(32, 16, kernel_size=(3,2))
        self.conv_tran0 = nn.ConvTranspose3d(16, image_channels, kernel_size=(2,3))

        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x):
        print("encode")
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
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z, mu, logvar, indices1, indices2, indices3, indices4

    def decode(self, z, indices1, indices2, indices3, indices4):
        print("decode", z.shape)
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
#model.load_state_dict(torch.load('c3dcnnvae.torch', map_location='cpu'))

#loss function and optimizer
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=OPT_LEARN_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = STEP_SIZE, gamma = GAMMA)

#load dataset
trainset = CustomDataset(train_dir)
testset = CustomDataset(test_dir)

train_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True,  **kwargs)
test_loader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False,  **kwargs)

#train and test model
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,_) in enumerate(train_loader):
        print("starting training")
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
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)
#        with torch.no_grad():
#            sample = Variable(torch.randn(64, ZDIMS)) ###edit parameters
#            sample = model.decode(sample).cpu()
#            save_image(sample.view(1, 61, 73, 61), ###edit parameters
#                       'results/sample_' + str(epoch) + '.png')

#save model state
torch.save(model.state_dict(), '3dcnnvae.torch')
