#!/usr/bin/env python3

'''
@author: d. lussier

Splits processed ABIDE data into train-test data and organizes 
into labeled classification folders for use by dataloader.
'''

import os
from glob import glob
from tqdm import tqdm
from shutil import copyfile, move
from sklearn.model_selection import train_test_split

#designate filepaths
root_dir = '../'
data_dir = os.path.join(root_dir,'output_abide_connectome/')
train_dir = os.path.join(root_dir,'vae_model/train/')
test_dir = os.path.join(root_dir,'vae_model/test/')
pitt_dir = os.path.join(data_dir,'pitt/rmap_seeds/')
olin_dir = os.path.join(data_dir,'olin/rmap_seeds/')
sdsu_dir = os.path.join(data_dir,'sdsu/rmap_seeds/')
trinity_dir = os.path.join(data_dir,'trinity/rmap_seeds/')
usm_dir = os.path.join(data_dir,'usm/rmap_seeds/')
yale_dir = os.path.join(data_dir,'yale/rmap_seeds/')
leuven_1_dir = os.path.join(data_dir,'leuven_1/rmap_seeds/')
leuven_2_dir = os.path.join(data_dir,'leuven_2/rmap_seeds/')
kki_dir = os.path.join(data_dir,'kki/rmap_seeds/')
nyu_dir = os.path.join(data_dir,'nyu/rmap_seeds/')
ucla_1_dir = os.path.join(data_dir,'ucla_1/rmap_seeds/')
ucla_2_dir = os.path.join(data_dir,'ucla_2/rmap_seeds/')
maxmun_dir = os.path.join(data_dir,'maxmun/rmap_seeds/')
caltech_dir = os.path.join(data_dir,'caltech/rmap_seeds/')
sbl_dir = os.path.join(data_dir,'sbl/rmap_seeds/')
#ohsu_dir = os.path.join(data_dir,'ohsu/rmap_seeds/')
#um_1_dir = os.path.join(data_dir,'um_1/rmap_seeds/')
#um_2_dir = os.path.join(data_dir,'um_2/rmap_seeds/')
#cmu_dir = os.path.join(data_dir,'cmu/rmap_seeds/')
#stanford_dir = os.path.join(data_dir,'stanfordrmap_seeds/')

#separate train and test data for each site
for site_dir in [pitt_dir,olin_dir,sdsu_dir,trinity_dir,usm_dir,
          yale_dir,leuven_1_dir,leuven_2_dir,kki_dir,nyu_dir,
          ucla_1_dir,ucla_2_dir,maxmun_dir,caltech_dir,sbl_dir
#          ohsu_dir,um_1_dir,um_2_dir,cmu_dir,stanford_dir
          ]:

    #create train and test directories within the site directory
    train_data = os.path.join(site_dir,'train/')
    test_data = os.path.join(site_dir,'test/')
    for path in [train_dir,test_dir]:
        if not os.path.exists(path):
            os.mkdir(path)
    
    #randomize and split training and test data for the site
    all_files = glob(os.path.join(site_dir,"rmap_*.nii.gz"))

    train,test = train_test_split(all_files,test_size = 0.2,random_state = 12345, shuffle=True)

    for t in tqdm(train):
        copyfile(t,os.path.join(train_data,os.path.split(t)[1]))
    
    for t in tqdm(test):
        copyfile(t,os.path.join(test_data,os.path.split(t)[1]))
    
    if site_dir == pitt_dir:
        label = '01'
    
    elif site_dir == olin_dir:
        label = '02'

    elif site_dir == sdsu_dir:
        label = '03'
    
    elif site_dir == trinity_dir:
        label = '04'
    
    elif site_dir == usm_dir:
        label = '05'
    
    elif site_dir == yale_dir:
        label = '06'
    
    elif site_dir == leuven_1_dir:
        label = '07'
    
    elif site_dir == leuven_2_dir:
        label = '07'
    
    elif site_dir == kki_dir:
        label = '08'
    
    elif site_dir == nyu_dir:
        label = '09'
    
    elif site_dir == ucla_1_dir:
        label = '10'
    
    elif site_dir == ucla_2_dir:
        label = '10'
    
    elif site_dir == maxmun_dir:
        label = '11'

   
    elif site_dir == caltech_dir:
        label = '12'
    
    elif site_dir == sbl_dir:
        label = '13'

#    elif site_dir == ohsu_dir:
#        label = '03'

#    elif site_dir == um_1_dir:
#        label = '06'
#    
#    elif site_dir == um_2_dir:
#        label = '06'

#    elif site_dir == cmu_dir:
#        label = '09'
    
#    elif site_dir == stanford_dir:
#        label = '13'
    
    #create training root directory  for site and move training data into it
    train_files = glob(os.path.join(train_data,"*.nii.gz"))
    site_train_dir = os.path.join(train_dir,label)
    for path in [site_train_dir]:
        if not os.path.exists(path):
            os.mkdir(path)
    for file in train_files:
        move(file, site_train_dir)
    
    #create test root directory for site and move test data into it
    test_files = glob(os.path.join(test_data,"*.nii.gz"))
    site_test_dir = os.path.join(test_dir,label)
    for path in [site_test_dir]:
        if not os.path.exists(path):
            os.mkdir(path)
    for file in test_files:
        move(file, site_test_dir)
