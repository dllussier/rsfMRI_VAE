import torch
import os
import re
import pandas as pd
import numpy as np
import torch.utils.data
from glob import glob
from tqdm import tqdm
from shutil import copyfile
from nilearn import datasets, masking
from nibabel import load as load_fmri
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from nilearn.input_data import NiftiMasker

CUDA = True
SEED = 1
BATCH_SIZE = 1
LOG_INTERVAL = 10
EPOCHS = 50
ZDIMS = 50
CLASSES = 10 #to be added
LEARN_RATE = 1e-4
STEP_SIZE = 1 #to be added
GAMMA = 0.9 #to be added


torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

#import dataset
data = datasets.fetch_abide_pcp(derivatives=['func_preproc'], #'rois_cc200', 'func_mask'],
                        n_subjects=5)

func = data.func_preproc #4D data
#target_func = data.rois_cc200
#func_mask = data.func_mask

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' % #location of image
      func[0])  
print(data.keys())

#fetching and processing list of names of files
###to dos: assign site classification, remove site id from subject name
#input_file = "/home/lussier/nilearn_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
#f = pd.read_csv(input_file, header = 0,  sep=',')

#f.to_csv("/home/lussier/nilearn_data/ABIDE_pcp/metadata.csv")
#df = pd.read_csv('/home/lussier/nilearn_data/ABIDE_pcp/metadata.csv', skipinitialspace=True, usecols = ['FILE_ID'])

#generate list of filenames
#names = df.FILE_ID.tolist()
#n_files = len(names)
    
#mask data, extract volumes, save with no site id, convert to np array/torch tensors
saving_dir  = './data/volumes/'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

for idx in tqdm(range(len(func))):
    func_data = func[idx]
    sub_name = re.findall(r'_\d',func_data)[0] #edit to remove site names but keep numbers 

    #mask volumes
    epi_mask = masking.compute_epi_mask(func)
    masker = NiftiMasker(mask_img=epi_mask) 
    BOLD = masker.fit_transform(func_data) 
    
    #designate timepoints for volumes
    timepoints = np.arange(start = 0, stop = 400, step = 2)[:BOLD.shape[0]]
    df = pd.DataFrame()
    df['timepoints'] = timepoints
    
    #extract individual volumes
    trial_start = np.arange(start = 0, stop = timepoints.max(), step = 1)
    interest_start = trial_start + 0
    interest_stop = trial_start + 2
    
    temp = []
    for time in timepoints:
        if any([np.logical_and(interval[0] <= time,time <= interval[1]) for interval in zip(interest_start,interest_stop)]):
            temp.append(1)
        else:
            temp.append(0)
    df['volumes'] = temp
    idx_picked = list(df[df['volumes'] == 1].index)
    BOLD_picked = BOLD[idx_picked]

    #save volumes as 3d samples    
    for ii,sample in enumerate(BOLD_picked):
        back_to_3D  = masker.inverse_transform(sample)
        saving_name = os.path.join(saving_dir,
                                   f'{sub_name}_volume{ii+1}.nii.gz')
        back_to_3D.to_filename(saving_name)

#randomize and split training and test data 
volumes_dir = './data/volumes/' 
train_dir = './data/volumes/train/'
test_dir = './data/volumes/test/'
for d in [train_dir,test_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

all_files = glob(os.path.join(volumes_dir,"*.nii.gz"))

train,test = train_test_split(all_files,test_size = 0.2,random_state = 12345, shuffle=True)

for f in tqdm(train):
    copyfile(f,os.path.join(train_dir,f.split('/')[-1]))
    
for f in tqdm(test):
    copyfile(f,os.path.join(test_dir,f.split('/')[-1]))

# load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

#define model
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

###add pooling
### missing missing channel deminsion
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

optimizer = optim.Adam(model.parameters(), lr=OPT_LEARN_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = STEP_SIZE, gamma = GAMMA)

#custom dataset for loader 
class CustomDataset(Dataset):    
    def __init__(self,data_root):
        self.samples = []       
        for item in glob(os.path.join(data_root,'*.nii.gz')):
            self.samples.append(item)
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        temp = load_fmri(self.samples[idx]).get_data()
        max_weight = temp.max()
        temp = temp / max_weight
        min_weight = np.abs(temp.min())
        temp = temp + min_weight
        return temp,max_weight,min_weight

#load dataset
trainset = CustomDataset(train_dir)
testset = CustomDataset(test_dir)

train_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

#train and test model
def train(epoch):
    model.train()
    train_loss = 0
    for idx, (data, _, _) in enumerate(train_loader):
        data = Variable(data)
        if CUDA:
            data = data.cuda()
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
    for i, (data, _, _) in enumerate(test_loader):
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
                     './results/reconstruction_' + str(epoch) + '.png', nrow=n)
        '''        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
###parameters need editing
if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        scheduler.step()
        train(epoch)
        test(epoch)
#        with torch.no_grad():
#            sample = Variable(torch.randn(64, ZDIMS)) ###edit parameters
#            sample = model.decode(sample).cpu()
#            save_image(sample.view(64, 1, 28, 28), ###edit parameters
#                       'results/sample_' + str(epoch) + '.png')

#save model state
torch.save(model.state_dict(), 'cnnvae.torch')
