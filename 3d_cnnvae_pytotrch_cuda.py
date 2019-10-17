import torch
import os
import re
import shutil
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
from nilearn.image import resample_img
from nilearn.input_data import NiftiMasker

CUDA = True
SEED = 1
BATCH_SIZE = 1
LOG_INTERVAL = 10
EPOCHS = 50
ZDIMS = 50
CLASSES = 2#0 
OPT_LEARN_RATE = 1e-4
STEP_SIZE = 1 
GAMMA = 0.9 
HDIM=1024

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

#import dataset
data = datasets.fetch_abide_pcp(derivatives=['func_preproc'], n_subjects=50)

func = data.func_preproc #4D data

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' % #location of image
      func[0])  
print(data.keys())

#move functional data to local data directory
for f in func:
    shutil.move(f, './data/')

#randomize and split training and test data 
func_dir = './data/'
train_dir = './data/train/'
test_dir = './data/test/'
for p in [train_dir,test_dir]:
    if not os.path.exists(p):
        os.mkdir(p)

all_files = glob(os.path.join(func_dir,"*.nii.gz"))

train,test = train_test_split(all_files,test_size = 0.2,random_state = 12345, shuffle=True)

for t in tqdm(train):
    copyfile(t,os.path.join(train_dir,os.path.split(t)[1]))
    
for t in tqdm(test):
    copyfile(t,os.path.join(test_dir,os.path.split(t)[1]))

#move data into respective site id label folders
pitt_dir = './data/train/pitt/'
olin_dir = './data/train/olin/'
ohsu_dir = './data/train/ohsu/'
sdsu_dir = './data/train/sdsu/'
trinity_dir = './data/train/trinity/'
um_1_dir = './data/train/um_1/'
um_2_dir = './data/train/um_2/'
usm_dir = './data/train/usm/'
yale_dir = './data/train/yale/'
cmu_dir = './data/train/cmu/'
leuven_1_dir = './data/train/leuven_1/'
leuven_2_dir = './data/train/leuven_2/'
kki_dir = './data/train/kki/'
nyu_dir = './data/train/nyu/'
stanford_dir = './data/train/stanford'
ucla_1_dir = './data/train/ucla_1/'
ucla_2_dir = './data/train/ucla_2/'
maxmun_dir = './data/train/maxmun/'
caltech_dir = './data/train/caltech/'
sbl_dir = './data/train/sbl/'
pitt_test_dir = './data/test/pitt/'
olin_test_dir = './data/test/olin/'
ohsu_test_dir = './data/test/ohsu/'
sdsu_test_dir = './data/test/sdsu/'
trinity_test_dir = './data/test/trinity/'
um_1_test_dir = './data/test/um_1/'
um_2_test_dir = './data/test/um_2/'
usm_test_dir = './data/test/usm/'
yale_test_dir = './data/test/yale/'
cmu_test_dir = './data/test/cmu/'
leuven_1_test_dir = './data/test/leuven_1/'
leuven_2_test_dir = './data/test/leuven_2/'
kki_test_dir = './data/test/kki/'
nyu_test_dir = './data/test/nyu/'
stanford_test_dir = './data/test/stanford'
ucla_1_test_dir = './data/test/ucla_1/'
ucla_2_test_dir = './data/test/ucla_2/'
maxmun_test_dir = './data/test/maxmun/'
caltech_test_dir = './data/test/caltech/'
sbl_test_dir = './data/test/sbl/'

for c in [pitt_dir,olin_dir,ohsu_dir,sdsu_dir,trinity_dir,um_1_dir,um_2_dir,
          usm_dir,yale_dir,cmu_dir,leuven_1_dir,leuven_2_dir,kki_dir,nyu_dir,
          stanford_dir,ucla_1_dir,ucla_2_dir,maxmun_dir,caltech_dir,sbl_dir,
          pitt_test_dir,olin_test_dir,ohsu_test_dir,sdsu_test_dir,
          trinity_test_dir,um_1_test_dir,um_2_test_dir,usm_test_dir,
          yale_test_dir,cmu_test_dir,leuven_1_test_dir,leuven_2_test_dir,
          kki_test_dir,nyu_test_dir,stanford_test_dir,ucla_1_test_dir,
          ucla_2_test_dir,maxmun_test_dir,caltech_test_dir, sbl_test_dir]:
    if not os.path.exists(c):
        os.mkdir(c)

train_files = glob(os.path.join(train_dir,"*.nii.gz"))    
for f in train_files:  
    if "Pitt" in f:
        shutil.move(f, pitt_dir)       
    if "Olin" in f: 
        shutil.move(f, olin_dir)            
    if "OHSU" in f: 
        shutil.move(f, ohsu_dir)
    if "SDSU" in f: 
        shutil.move(f, sdsu_dir)
    if "Trinity" in f: 
        shutil.move(f, trinity_dir)
    if "UM_1" in f: 
        shutil.move(f, um_1_dir)
    if "UM_2" in f: 
        shutil.move(f, um_2_dir)    
    if "USM" in f: 
        shutil.move(f, usm_dir)
    if "Yale" in f: 
        shutil.move(f, yale_dir)
    if "CMU" in f: 
        shutil.move(f, cmu_dir)        
    if "Leuven_1" in f: 
        shutil.move(f, leuven_1_dir)
    if "Leuven_2" in f: 
        shutil.move(f, leuven_2_dir)
    if "KKI" in f: 
        shutil.move(f, kki_dir)
    if "NYU" in f: 
        shutil.move(f, nyu_dir)
    if "Stanford" in f: 
        shutil.move(f, stanford_dir) 
    if "UCLA_1" in f: 
        shutil.move(f, ucla_1_dir)
    if "UCLA_2" in f: 
        shutil.move(f, ucla_2_dir)
    if "MaxMun" in f:
        shutil.move(f, maxmun_dir)        
    if "Caltech" in f: 
        shutil.move(f, caltech_dir)
    if "SBL" in f: 
        shutil.move(f, sbl_dir)

test_files = glob(os.path.join(test_dir,"*.nii.gz"))    
for f in test_files:  
    if "Pitt" in f:
        shutil.move(f, pitt_test_dir)       
    if "Olin" in f: 
        shutil.move(f, olin_test_dir)            
    if "OHSU" in f: 
        shutil.move(f, ohsu_test_dir)
    if "SDSU" in f: 
        shutil.move(f, sdsu_test_dir)
    if "Trinity" in f: 
        shutil.move(f, trinity_test_dir)
    if "UM_1" in f: 
        shutil.move(f, um_1_test_dir)
    if "UM_2" in f: 
        shutil.move(f, um_2_test_dir)    
    if "USM" in f: 
        shutil.move(f, usm_test_dir)
    if "Yale" in f: 
        shutil.move(f, yale_test_dir)
    if "CMU" in f: 
        shutil.move(f, cmu_test_dir)        
    if "Leuven_1" in f: 
        shutil.move(f, leuven_1_test_dir)
    if "Leuven_2" in f: 
        shutil.move(f, leuven_2_test_dir)
    if "KKI" in f: 
        shutil.move(f, kki_test_dir)
    if "NYU" in f: 
        shutil.move(f, nyu_test_dir)
    if "Stanford" in f: 
        shutil.move(f, stanford_test_dir) 
    if "UCLA_1" in f: 
        shutil.move(f, ucla_1_test_dir)
    if "UCLA_2" in f: 
        shutil.move(f, ucla_2_test_dir)
    if "MaxMun" in f:
        shutil.move(f, maxmun_test_dir)        
    if "Caltech" in f: 
        shutil.move(f, caltech_test_dir)
    if "SBL" in f: 
        shutil.move(f, sbl_test_dir)
        
#mask data, extract volumes, save with no site id, convert to np array/torch tensors
for s in [pitt_dir,olin_dir,ohsu_dir,sdsu_dir,trinity_dir,um_1_dir,um_2_dir,
          usm_dir,yale_dir,cmu_dir,leuven_1_dir,leuven_2_dir,kki_dir,nyu_dir,
          stanford_dir,ucla_1_dir,ucla_2_dir,maxmun_dir,caltech_dir,sbl_dir,
          pitt_test_dir,olin_test_dir,ohsu_test_dir,sdsu_test_dir,
          trinity_test_dir,um_1_test_dir,um_2_test_dir,usm_test_dir,
          yale_test_dir,cmu_test_dir,leuven_1_test_dir,leuven_2_test_dir,
          kki_test_dir,nyu_test_dir,stanford_test_dir,ucla_1_test_dir,
          ucla_2_test_dir,maxmun_test_dir,caltech_test_dir, sbl_test_dir]:
    volumes_dir  = os.path.join(s,'volumes/')
    func_files = glob(os.path.join(s,"*_func_preproc.nii.gz"))
    if not os.path.exists(volumes_dir):
        os.mkdir(volumes_dir)
    print(volumes_dir)
    
    for idx in tqdm(range(len(func_files))):
        func_data = func_files[idx]
        sub_name = re.findall(r'_\d+',func_data)[0] #remove site from subject names but keep numbers 
        #if len(sub_name) == 0:
        #    break
        #else:
        #    sub_name=sub_name[0]
    
        #mask volumes
        epi_mask = masking.compute_epi_mask(func_files)
        masker = NiftiMasker(mask_img=epi_mask) 
        BOLD = masker.fit_transform(func_data) 
        
        #designate timepoints for volumes
        timepoints = np.arange(start = 0, stop = 400, step = 2)[:BOLD.shape[0]]
        df = pd.DataFrame()
        df['timepoints'] = timepoints
        
        #extract individual volumes
        trial_start = np.arange(start = 0, stop = timepoints.max(), step = 1)
        interest_start = trial_start + 0
        interest_stop = trial_start + 1
        
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
            saving_name = os.path.join(volumes_dir,
                                       f'{sub_name}_volume{ii+1}.nii.gz')
            back_to_3D.to_filename(saving_name)
    
    #remove original 4D files
    for f in func_files:      
        os.remove(f)
    
    #move 3D nifti files to site label folder
    volume_files = glob(os.path.join(volumes_dir,"*.nii.gz"))
    for v in volume_files: 
        shutil.move(v, s)
    
    #delete empty volumes folder
    #os.rmdir(volumes_dir)

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
    def __init__(self, image_channels=3, h_dim=HDIM, z_dim=ZDIMS, n_classes=CLASSES):
        super(CNNVAE, self).__init__()
        
        print("CNNVAE")
        #encoder cnn layers
        self.encoder = nn.Sequential(
            nn.Conv3d(image_channels, 16, kernel_size=3, stride=2), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=4, stride=2), #padding=0, dilation=1, return_indices=False, ceil_mode=False),
            nn.Conv3d(16, 32, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=4, stride=2),
            nn.Conv3d(32, 32, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=4, stride=2),
            Flatten()
        )
        #nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.fc1 = nn.Linear(h_dim, z_dim) #mu
        self.fc2 = nn.Linear(h_dim, z_dim) #logvar
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        #decoder cnn layers
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.MaxUnpool3d(kernel_size=5, stride=2, padding=0),
            nn.ConvTranspose3d(h_dim, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxUnpool3d(kernel_size=5, stride=2, padding=0), ##check order of layers
            nn.ConvTranspose3d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxUnpool3d(kernel_size=5, stride=2, padding=0),
            nn.ConvTranspose3d(32, 16, kernel_size=6, stride=2),
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
        print("reparameterize")
    def bottleneck(self, h: Variable) -> (Variable, Variable):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        print("bottleneck")
    def encode(self, x: Variable) -> (Variable, Variable):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar
        print("encode")
    def decode(self, z: Variable) -> Variable:
        z = self.fc3(z)
        z = self.decoder(z)
        return z
        print("decode")
    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
        print("forward")
model = CNNVAE()
if CUDA:
    model.cuda()

#load previous state   
#model.load_state_dict(torch.load('cnnvae.torch', map_location='cpu'))

#loss function and optimizer
def loss_function(recon_x, x, mu, logvar) -> Variable:
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

optimizer = optim.Adam(model.parameters(), lr=OPT_LEARN_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = STEP_SIZE, gamma = GAMMA)

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
        return load, label
'''
https://discuss.pytorch.org/t/how-to-load-nib-to-pytorch/40947
import nibabel as nib#http://nipy.org/nibabel/gettingstarted.html
class Dataloder_img(data.Dataset):
    def __init__(self,data_root,site_dir,transforms):
        self.data_root = data_root
        self.site_dir = site_dir
        self.transforms = transforms
        self.files = os.listdir(self.data_root)
        self.labels = os.listdir(self.site_dir)
        print(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,idx):
        img_name = self.files[idx]
        label_name = self.labels[idx]
        img = load_fmri(os.path.join(self.data_root,img_name)) #!Image.open(os.path.join(self.data_root,img_name))
        #change to numpy
        img = np.array(img.dataobj)
        #change to PIL 
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        
        print(img.size)
        
        label = load_fmri(os.path.join(self.site_dir,label_name))#!Image.open(os.path.join(self.site_dir,label_name))
        #change to numpy
        label = np.array(label.dataobj)
        #change to PIL 
        label = Image.fromarray(label.astype('uint8'), 'RGB')
        
        print(label.size)
        
        if self.transforms:
            img = self.transforms(img)
            label = self.transforms(label)
            return img,label
        else:
            return img, label
full_dataset = Dataloder_img(' image ',' labels ',tfms.Compose([tfms.ToTensor()]))#
'''
 
#load dataset
trainset = CustomDataset(train_dir)
testset = CustomDataset(test_dir)

train_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

#train and test model
def train(epoch):
    model.train()
    train_loss = 0
    for idx, (data, _) in enumerate(train_loader):
        print("starting training")
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
