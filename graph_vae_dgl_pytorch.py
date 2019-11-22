#!/usr/bin/env python3

import torch
import os
import dgl
import torch.utils.data
import numpy as np
import networkx as nx
from glob import glob
import dgl.function as fn
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

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
    
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

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
        G = nx.MultiGraph()
        #load numpy array from saved .npy file
        a = np.load(name)
        #reshape stacked numpy array to 2d 
        b = np.reshape(a, (39,39), order='C')        
        #convert reshaped numpy array to networkx graph 
        D = nx.nx.convert.to_networkx_graph(b, create_using=nx.MultiGraph)
        G.add_nodes_from(D.nodes)
        G.add_edges_from(D.edges) 
        #convert netowrkx graph to dgl graph
        graph=dgl.DGLGraph()
        graph.from_networkx(G)
        return graph, label
    
#create custom collate funtion
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels=np.asarray(labels, dtype='float')
    return batched_graph, torch.Tensor(labels)

#sends a message of node feature h
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

#vae using gcn
class VAE(nn.Module):
    def __init__(self, g_dim, h_dim1, h_dim2, z_dim, n_classes):
        super(VAE, self).__init__()
        
        # encoder
        self.fc1 = GCN(g_dim, h_dim1, F.relu)
        self.fc2 = GCN(h_dim1, h_dim2, F.relu)
        self.fc31 = GCN(h_dim2, z_dim, F.linear) #mu
        self.fc32 = GCN(h_dim2, z_dim, F.linear) #logvar
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
        h = g.in_degrees().view(-1, 1).float()
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        mu, log_var = self.encoder(hg)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

model = VAE(g_dim=GDIM, h_dim1= HDIM1, h_dim2=HDIM2, z_dim=ZDIMS, n_classes=NSITES)
if CUDA:
    model.cuda()

#loss function
def loss_function(recon_g, g, mu, log_var):
    BCE = F.binary_cross_entropy(recon_g, g.view(-1, 1521), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)

#load data
train_dir = './data01/train/'
test_dir = './data01/test/'

trainset = CustomDataset(train_dir)
testset = CustomDataset(test_dir)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, **kwargs)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, **kwargs)

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
