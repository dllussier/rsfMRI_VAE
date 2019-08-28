#!/usr/bin/env python3

import torch
import dgl
import numpy as np
import networkx as nx

import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim
import dgl.function as fn

from torch import nn
from torch.nn import functional as F
from scipy.misc import imread
#from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# changed configuration to this instead of argparse for easier interaction
CUDA = True
SEED = 1
BATCH_SIZE = 32
LOG_INTERVAL = 10
EPOCHS = 20
ZDIMS = 20

cuda = torch.device('cuda')

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)
    
#load dataloader instances directly into gpu memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

g = dgl.MultiGraph()
graph, label = data[0]
fig, ax = plt.subplots()
nx.draw(g.to_networkx(), ax=ax)
ax.set_title('Class: {:d}'.format(label))
plt.show()

#sends message of node feature h
def gcn_msg(edge):
    msg = edge.src['h'] * edge.src['norm']
    return {'m': msg}

def reduce(node):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
    return {'h': accum}

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

#graph convolution
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):   
        g.ndata['h'] = feature  # Initialize the node features with h.
        g.update_all(gcn_msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


#vae using gcn
class VAE(nn.Module):
    def __init__(self, g_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder
        self.fc1 = GCN(g_dim, h_dim1, F.relu)
        self.fc2 = GCN(h_dim1, h_dim2, F.relu)
        self.fc31 = GCN(h_dim2, z_dim, F.linear)
        self.fc32 = GCN(h_dim2, z_dim, F.linear)
        # decoder
        self.fc4 = GCN(z_dim, h_dim2, F.relu)
        self.fc5 = GCN(h_dim2, h_dim1, F.relu)
        self.fc6 = GCN(h_dim1, g_dim, F.sigmoid)
        
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

vae = VAE(g_dim=784, h_dim1= 512, h_dim2=256, z_dim=ZDIMS)   

model = vae
if CUDA:
    model.cuda()

#load data
'''
def read_dataset(path):
    images_path = f"{path}/images"
    labels_path = f"{path}/labels"

    images = np.zeros((61, 496, 768)) #edit these matrices
    labels1 = np.zeros((61, 496, 768))
    labels2 = np.zeros((61, 496, 768))
    labels3 = np.zeros((61, 496, 768))
    
    for i in range(61):
        img_file_path = f"{images_path}/images_{i}.png"
        lbl_file_path1 = f"{labels_path}/class1/_{i}.png"
        lbl_file_path2 = f"{labels_path}/class2/_{i}.png"
        lbl_file_path3 = f"{labels_path}/class17/_{i}.png"
        
        images[i] = imread(img_file_path)
        labels1[i] = imread(lbl_file_path1)
        labels2[i] = imread(lbl_file_path2)
        labels3[i] = imread(lbl_file_path3)
        return images,labels1 ,labels2,labels3

#ref: https://pytorch.org/docs/stable/data.html
#ref: https://towardsdatascience.com/how-to-build-custom-dataloader-for-your-own-dataset-ae7fd9a40be6        
        
images, labels1,labels2,labels3 = read_dataset("{filepath}/train")

testimages, mask1 ,mask2,mask3 = read_dataset("{filepath}/test")
'''

trainset = read_dataset("{filepath}/train")
testset = read_dataset("{filepath}/test")

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=None, shuffle=True, **kwargs)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, collate_fn=None, shuffle=True, **kwargs)

#loss function
def loss_function(recon_g, g, mu, log_var):
    BCE = F.binary_cross_entropy(recon_g, g.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)

#train and test
def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(cuda)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(cuda)
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, EPOCHS):
    train(epoch)
    test()

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(cuda)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
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
    
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(cuda)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).data
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                  recon_batch.view(BATCH_SIZE, 2, 28, 28)[:n]]) #edit these parameters
                save_image(comparison.data.cpu(),
                           '/home/lussier/fMRI_VQ_VAE/results/practice/dglreconstruction_' + str(epoch) + '.png', nrow=n)
          
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
            save_image(sample.data.view(BATCH_SIZE, 2, 28, 28),
                       '/home/lussier/fMRI_VQ_VAE/results/practice/dglsample_' + str(epoch) + '.png')
