# torch stuff
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

# torchsummary and torchvision
#from torchsummary import summary
#from torchvision.utils import save_image

# matplotlib stuff
import matplotlib.pyplot as plt
import matplotlib.image as img

# numpy and pandas
import numpy as np
import pandas as pd

# Common python packages
import datetime
import os
import sys
import time
from train import train_cgan
from utils import *


# Parameters for Adam optimizer
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
# Parameters for split
n_splits = 1
split = None
# Parameters of input
tsv_path = '/export/home/cse180022/apprimage_simo/local_image_processing/image_synthesis/subjects_list/tier_1_labeled_paired_gaudo.tsv'
diagnoses = ['gaudo_0', 'gaudo_1'] ## to change ## diagnoses will be gado, not_gado
input_dir = '/export/home/cse180022/apprimage_simo/image_preprocessing_data/ds9_caps'
# Create dataloaders
batch_size = 2
baseline = 'False'
# Parameter for operations on dataloader
mode = 'image'
preprocessing = 't1-linear'
num_workers = 2 #n_proc
print('done import')


transformations = get_transforms(mode, minmaxnormalization=True)

if split is None:
    fold_iterator = range(n_splits)
else:
    fold_iterator = split

for fi in fold_iterator:
    training_df, valid_df = load_data(
        tsv_path,
        diagnoses,
        fi,
        n_splits=n_splits,
        baseline=baseline)

    data_train = MRIDatasetImage(
        input_dir,
        training_df,
        preprocessing,
        transformations=transformations)
    print('done data train')
    data_valid = MRIDatasetImage(
        input_dir,
        training_df,
        preprocessing,
        transformations=transformations)
    print('done data valid')

    # Use argument load to distinguish training and testing

    #### insert here data sampler
    # data_sampler = generate_sampler(data_train, 'weighted')

    train_loader = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    print('loader train')

    valid_loader = DataLoader(
        data_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print('loader valid')


# Number of epochs
num_epoch = 3

# Train the generator
#generator = train_cgan(train_loader, valid_loader, num_epoch=3,
#                            lr=lr, beta1=beta1, beta2=beta2)
