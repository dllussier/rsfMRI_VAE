#!/usr/bin/env python3

import torch
import os
import re
import pandas as pd
import numpy as np
import torch.utils.data
from glob import glob
from tqdm import tqdm
from shutil import copyfile
from nilearn import datasets
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_img


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

#import atlas
atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']
# Loading atlas data stored in 'labels'
labels = atlas['labels']

#import dataset
data = datasets.fetch_abide_pcp(derivatives=['func_preproc','rois_cc200', 'func_mask'],
                        n_subjects=10)

func = data.func_preproc #4D data
target_func = data.func_mask
reshaped = data.func_preproc_reshaped
resample = data.func_preproc_resample
mask_reshaped = data.mask_reshaped

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' % #location of image
      func[0])  
print(data.keys())

#fetching and processing list of names of files
###to dos: assign site classification, remove site id from subject name, randomize list
input_file = "/home/lussier/nilearn_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
f = pd.read_csv(input_file, header = 0,  sep=',')

f.to_csv("/home/lussier/nilearn_data/ABIDE_pcp/metadata.csv")
df = pd.read_csv('/home/lussier/nilearn_data/ABIDE_pcp/metadata.csv', skipinitialspace=True, usecols = ['FILE_ID'])

#generate list of filenames
names = df.FILE_ID.tolist()
n_files = len(names)

#resize and reshape images
for idx in range(len(func)):
    ###resample for (voxel_size = (2.386364,2.386364,2.4))
    resampled = resample_img(func,
                             target_affine  = target_func.affine, #check this to see if parcellation is better
                             target_shape   = (88,88,66))
    resampled.to_filename('func_preproc_reshaped.nii.gz')   
    
#mask data, extract volumes, save with no site id, convert to np array/torch tensors
saving_dir  = './data/volumes/'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

for idx in tqdm(range(len(reshaped))):
    picked_data = reshaped[idx]
    sub_name = re.findall(r'CSI\d',picked_data)[0]
    n_session = int(re.findall(r'\d+',re.findall(r'Sess-\d+_',picked_data)[0])[0])
    n_run = int(re.findall(r'\d+',re.findall(r'Run-\d+',picked_data)[0])[0])
    picked_data_mask = target_func

    ###resize mask
    
    #reshape mask
    resampled = resample_img(picked_data_mask,
                             target_affine = target_func.affine, #check this for proper target
                             target_shape = (88,88,66))
    resampled.to_filename('mask_reshaped.nii.gz')

    ###binarize mask

    #mask volumes
    masker = NiftiMasker(mask_img = mask_reshaped)
    BOLD = masker.fit_transform(picked_data)
    timepoints = np.arange(start = 0,stop = 400,step = 2)[:BOLD.shape[0]]
    df = pd.DataFrame()
    df['timepoints'] = timepoints

    ###extract and rename individual volumes


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

        #encoder cnn layers
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

#split training and test data 
####edit this to pull from list of randomized subject ids that pull from folders
volumes_dir = '../data/volumes/' 
train_dir = '../data/train/'
test_dir = '../data/test/'
for d in [train_dir,test_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

all_files = glob(os.path.join(volumes_dir,"*.nii.gz"))

train,test = train_test_split(all_files,test_size = 0.2,random_state = 12345, shuffle=True)

for f in tqdm(train):
    copyfile(f,os.path.join(train_dir,f.split('/')[-1]))
    
for f in tqdm(test):
    copyfile(f,os.path.join(test_dir,f.split('/')[-1]))

#load data but do not reshuffle 
trainset = datasets.ImageFolder(root=train_dir)
testset = datasets.ImageFolder(root=test_dir)

train_loader = torch.utils.data.DataLoader(
        trainset('data', train=True, batch_size=BATCH_SIZE, 
                shuffle=False, **kwargs))

test_loader = torch.utils.data.DataLoader(
        testset('data', train=False, batch_size=BATCH_SIZE, 
                shuffle=False, **kwargs))

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
