
#!/usr/bin/env python3

import torch
import os
import torch.utils.data
from glob import glob
from tqdm import tqdm
from shutil import copyfile
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


CUDA = True
SEED = 1
BATCH_SIZE = 32
LOG_INTERVAL = 10
EPOCHS = 50
ZDIMS = 50
CLASSES = 10 #this needs added to the model

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

#split training and test data
volumes_dir = '../data/volumes/'
train_dir = '../data/train/'
test_dir = '../data/test/'
for d in [train_dir,test_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

all_files = glob(os.path.join(volumes_dir,"*.nii.gz"))

train,test = train_test_split(all_files,test_size = 0.2,random_state = 12345)

for f in tqdm(train):
    copyfile(f,os.path.join(train_dir,f.split('/')[-1]))
    
for f in tqdm(test):
    copyfile(f,os.path.join(test_dir,f.split('/')[-1]))

trainset = datasets.ImageFolder(root='../data/train/')
testset = datasets.ImageFolder(root='../data/test/')

# load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

#define model
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class CNNVAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=ZDIMS, n_classes=CLASSES):
        super(CNNVAE, self).__init__()

        #decoder cnn layers
        self.encoder = nn.Sequential(
            nn.Conv3d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim) #mu
        self.fc2 = nn.Linear(h_dim, z_dim) #logvar
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        #decoder cnn layers
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose3d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        if self.training:
            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
            return z
        else:
            return mu

    def bottleneck(self, h: Variable) -> (Variable, Variable):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def encode(self, x: Variable) -> (Variable, Variable):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z: Variable) -> Variable:
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

model = CNNVAE()
if CUDA:
    model.cuda()

#load previous state   
model.load_state_dict(torch.load('cnnvae.torch', map_location='cpu'))

#loss function and optimizer
def loss_function(recon_x, x, mu, logvar) -> Variable:
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)

#load data
train_loader = torch.utils.data.DataLoader(
        trainset('data', train=True, batch_size=BATCH_SIZE, 
                shuffle=True, **kwargs))

test_loader = torch.utils.data.DataLoader(
        testset('data', train=False, batch_size=BATCH_SIZE, 
                shuffle=True, **kwargs))

#train and test model
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if CUDA:
            data = data.cuda()
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
    for i, (data, _) in enumerate(test_loader):
        if CUDA:
            data = data.cuda()
        data = Variable(data, requires_grad=False)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data
        '''
        #this is to generate comparison images 
        #needs editing
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]]) #edit this
            save_image(comparison.data.cpu(),
                     '/home/lussier/fMRI_VQ_VAE/results/practice/reconstruction_' + str(epoch) + '.png', nrow=n)
        '''        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test(epoch)    
    sample = Variable(torch.randn(64, ZDIMS)) #edit parameters
    
    with torch.no_grad():
        sample = sample.cuda()   
        sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 28, 28), #edit these parameters
        '/home/lussier/fMRI_VQ_VAE/results/practice/sample_' + str(epoch) + '.png')

#save model state
torch.save(model.state_dict(), 'cnnvae.torch')
