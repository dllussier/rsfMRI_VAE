#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: desiree lussier

takes 3d nifti images in labeled folders 
"""

import torch
import os
import numpy as np
import torch.utils.data
from glob import glob
from nibabel import load as load_fmri
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

CUDA = True
SEED = 1
BATCH_SIZE = 1
LOG_INTERVAL = 10
EPOCHS = 50
ZDIMS = 50
CLASSES = 7 
OPT_LEARN_RATE = 1e-4
STEP_SIZE = 1 
GAMMA = 0.9 
HDIM=1024

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

train_dir = "./data01/train01/"
test_dir = "./data01/test01/"

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
        print('label is %s' % label)
        print('name is %s' % name)
        load = load_fmri(name).get_data()
        npimg = np.array(load)
        npimg_fit=(npimg +1)*127.5
        npimg_fit=npimg_fit.astype(np.uint8)
        transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                ]) 
        img=torch.tensor(transform(npimg_fit))
        nplabel=np.asarray(label, dtype='float')
        return img, nplabel 

#define model
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=HDIM):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=HDIM, z_dim=ZDIMS, n_classes=CLASSES):
        super(VAE, self).__init__()
        
        print("VAE")
        #encoder cnn layers
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=2),# stride=(1, 1)), padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),# stride=2, padding=0), #dilation=1, return_indices=False, ceil_mode=False),
            nn.Conv2d(16, 32, kernel_size=3),# stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),# stride=3, padding=0),
            nn.Conv2d(32, 96, kernel_size=2),# stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),# stride=2, padding=0),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim) #mu
        self.fc2 = nn.Linear(h_dim, z_dim) #logvar
        self.fc3 = nn.Linear(z_dim, h_dim)          #RuntimeError: size mismatch, m1: [1 x 96], m2: [1024 x 50] 
        
        #decoder cnn layers
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.MaxUnpool2d(kernel_size=5, stride=2, padding=0),
            nn.ConvTranspose2d(h_dim, 96, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxUnpool2d(kernel_size=5, stride=2, padding=0), 
            nn.ConvTranspose2d(96, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxUnpool2d(kernel_size=5, stride=2, padding=0),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        print("reparameterize") 
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        print("bottleneck") 
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar #RuntimeError: size mismatch, m1: [1 x 96], m2: [1024 x 50] at /pytorch/aten/src/THC/generic/THCTensorMathBlas.cu:268
    
    def encode(self, x):
        print("encode") 
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar 

    def decode(self, z):
        print("decode")
        z = self.fc3(z)
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        print("forward")
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
        
model = VAE()
if CUDA:
    model.cuda()
    
#load previous state   
#model.load_state_dict(torch.load('cnnvae.torch', map_location='cpu'))

#loss function and optimizer
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

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
    for idx, (data,_) in enumerate(train_loader):
        print("starting training")
        data = Variable(data)
        if CUDA:
            data = data.cuda()
        scheduler.step()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset),
                100. * idx / len(train_loader),
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
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data     
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
###parameters need editing
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
torch.save(model.state_dict(), 'cnnvae.torch')
