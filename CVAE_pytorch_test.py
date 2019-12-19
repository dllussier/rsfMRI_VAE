#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: desiree lussier

takes 3d nifti images in labeled folders 
"""

import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
#workaroundfor RuntimeError: expected backend CUDA and dtype Float but got backend CPU and dtype Float
torch.set_default_tensor_type(torch.cuda.FloatTensor)  
from glob import glob
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from nibabel import load as load_fmri

CUDA = True
SEED = 1
BATCH_SIZE = 1
LOG_INTERVAL = 10
EPOCHS = 5
ZDIMS = 10176
CLASSES = 7 
OPT_LEARN_RATE = 1e-4
STEP_SIZE = 1 
GAMMA = 0.9 
HDIM=576

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
        npimg = np.array(load, dtype='int32')
        npimg_fit=(npimg +1)*127.5
        npimg_fit=npimg_fit.astype(np.uint8)
        transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                ]) 
        img=torch.tensor(transform(npimg_fit))
        nplabel=np.asarray(label, dtype='int32')
        print('nplabel is %s' % nplabel)
        return img, nplabel 

#define model
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1) #x.view(x.size(0), 3*53*64*52)

class UnFlatten(nn.Module):
    def forward(self, input, size=HDIM):
        return input.view(input.size(0), size, 1, 1) #x.view(x.size(0), 3*53*64*52)

class Encoder(nn.Module):
    def __init__(self, image_channels=3, h_dim=HDIM, z_dim=ZDIMS, n_classes=CLASSES):
        super(Encoder, self).__init__()
        
        print("encoder")
        #encoder cnn layers
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(32, 96, kernel_size=2, stride=1)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=2, stride=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)
        
        self.flatten = Flatten()
        
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
    
    def forward(self, x):
        print("encode")
        h = F.relu(self.conv1(x))
        print("conv1")
        h, indices1 = self.maxpool(h)
        print("pool1")
        h = F.relu(self.conv2(h))
        print("conv2")
        h, indices2 = self.maxpool(h)
        print("pool2")        
        h = F.relu(self.conv3(h))
        print("conv3")
        h, indices3 = self.maxpool(h)
        print("pool3")
        h = F.relu(self.conv4(h))
        print("conv4")
        h, indices4 = self.maxpool(h)
        print("pool4")
        h = self.flatten(h)
        print("bottleneck") 
        mu, logvar = self.mu(h), self.logvar(h)
        print("reparameterize") 
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z, mu, logvar, indices1, indices2, indices3, indices4

class Decoder(nn.Module):
    def __init__(self, image_channels=3, h_dim=HDIM, z_dim=ZDIMS, n_classes=CLASSES):
        super(Decoder, self).__init__()
        
        print("decoder")
        #decoder cnn layers        
        self.latent = nn.Linear(z_dim, h_dim)  
        self.unflatten = UnFlatten()

        self.maxunpool = nn.MaxUnpool2d(kernel_size=2)
        
        self.conv_tran1 = nn.ConvTranspose2d(h_dim, 96, kernel_size=4, stride=1)
        self.conv_tran2 = nn.ConvTranspose2d(96, 32, kernel_size=4, stride=1)
        self.conv_tran3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=1)
        self.conv_tran4 = nn.ConvTranspose2d(16, image_channels, kernel_size=5, stride=1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z, indices1, indices2, indices3, indices4):
        print("decode")
        z = self.latent(z)
        z = self.unflatten(z)
        z = self.maxunpool(z, indices4)
        print("unpool4")
        z = F.relu(self.conv_tran1(z))
        print("deconv4")
        z = self.maxunpool(z, indices3)
        print("unpool3")
        z = F.relu(self.conv_tran2(z))
        print("deconv3")
        z = self.maxunpool(z, indices2)
        print("unpool2")
        z = F.relu(self.conv_tran3(z))
        print("deconv2")
        z = self.maxunpool(z, indices1)
        print("unpool1")
        z = F.relu(self.conv_tran4(z))
        print("deconv1")
        z = self.sigmoid(z)
        return z
#
#encoder = Encoder()
#decoder = Decoder() 

model = Encoder()
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

"""
def save_reconstructed_images(data, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            outputs.view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n, normalize=True)

"""
