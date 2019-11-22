#!/usr/bin/env python3

import torch
import os
import torch.utils.data
import numpy as np
import networkx as nx
from glob import glob
import argparse
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, VGAE

# changed configuration to this instead of argparse for easier interaction
CUDA = True
SEED = 1
BATCH_SIZE = 1
LOG_INTERVAL = 10
EPOCHS = 100
NSITES = 20 #number of sites
GDIM = 1521
HDIM1 = 1014
HDIM2 = 507
ZDIMS = 169

#load dataloader instances directly into gpu memory
cuda = torch.device('cuda')
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGAE')
args = parser.parse_args()
assert args.model in ['VGAE']
   
kwargs = {'num_workers': 1, 'pin_memory': True, 'VGAE': VGAE} if CUDA else {'VGAE': VGAE}

'''
#create customized dataset
class CustomDataset(Dataset):    
    def __init__(self,data_root):
        self.samples = []
        #self.transform = transforms.Compose([transforms.ToTensor()])
        
        for label in os.listdir(data_root):            
                labels_folder = os.path.join(data_root, label)

                for name in glob(os.path.join(labels_folder,'*.npy')):
                    self.samples.append((label,name)) 

        print('data root: %s' % data_root)
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label,name=self.samples[idx]
        print('label is %s' % label)
        print('name is %s' % name)
        graph = np.load(name)
        return graph, label
'''
    
#create custom collate funtion
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    labels=np.asarray(labels, dtype='float')
    graphs = torch.FloatTensor(torch.cat(graphs))
    labels = torch.LongTensor(torch.cat(labels).squeeze())
    return graphs, labels

#vae using gcn
class VAE(nn.Module):
    def __init__(self, g_dim, h_dim1, h_dim2, z_dim, n_classes):
        super(VAE, self).__init__()
        
        # encoder
        self.fc1 = GCNConv(g_dim, h_dim1, F.relu)
        self.fc2 = GCNConv(h_dim1, h_dim2, F.relu)
        self.fc31 = GCNConv(h_dim2, z_dim, F.linear)
        self.fc32 = GCNConv(h_dim2, z_dim, F.linear)
        # decoder
        self.fc4 = GCNConv(z_dim, h_dim2, F.relu)
        self.fc5 = GCNConv(h_dim2, h_dim1, F.relu)
        self.fc6 = GCNConv(h_dim1, g_dim, F.sigmoid)
        
    def encoder(self, g):
        h = self.fc1(g)
        h = self.fc2(h)
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = self.fc4(z)
        h = self.fc5(h)
        return self.fc6(h)
    
    def forward(self, g):
        mu, log_var = self.encoder(g)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

model = kwargs[args.model](VAE(g_dim=GDIM, h_dim1= HDIM1, h_dim2=HDIM2, z_dim=ZDIMS, n_classes=NSITES))
if CUDA:
    model.cuda()

#loss function
def loss_function(recon_g, g, mu, log_var):
    BCE = F.binary_cross_entropy(recon_g, g.view(-1, 1521), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)

#load data
train_dir = './data/train/'
test_dir = './data/test/'

trainset = CustomDataset(train_dir)
testset = CustomDataset(test_dir)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)#, **kwargs)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)#, **kwargs)

#train and test
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(cuda)
        optimizer.zero_grad()        
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(cuda)
            recon, mu, log_var = model(data)            
            test_loss += loss_function(recon, data, mu, log_var).item()        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)
        sample = torch.randn(BATCH_SIZE, ZDIMS)
        with torch.no_grad():
            sample = sample.to(cuda)   
            sample = model.decode(sample).cpu()
            #save_image(sample.data.view(BATCH_SIZE, 2, 39, 39),
#           '/home/lussier/fMRI_VQ_VAE/results/practice/dglsample_' + str(epoch) + '.png')
