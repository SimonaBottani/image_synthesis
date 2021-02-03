# torch stuff
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import argparse

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


parser = argparse.ArgumentParser(description='image synthesis')

parser.add_argument(
        'caps_dir',
        help='Data using CAPS structure.',
        default=None
    )
parser.add_argument(
        'output_results',
        help='Where results will be saved',
        default=None
    )

parser.add_argument(
        'tsv_path',
        help='TSV path with subjects/sessions to use for data generation.',
        default=None
    )

parser.add_argument(
        'n_epoch',
        type=int,
        default=200,
        help='number of epoch'
    )

parser.add_argument(
        'lr',
        type=float,
        default=0.0001,
        help='learning_rate'
    )

parser.add_argument(
    'beta1',
    type=float,
    default=0.5,
    help='beta1 for Adam Optimizer'
)

parser.add_argument(
    'beta2',
    type=float,
    default=0.999,
    help='beta1 for Adam Optimizer'
)

parser.add_argument(
    'n_splits',
    type=int,
    help='number of CV'
)


parser.add_argument(
    'batch_size',
    type=int,
    default=2,
    help='batch_size'
)


args = parser.parse_args()

## write command line arguments on json
commandline = parser.parse_known_args()
commandline_to_json(commandline)

## read command line arguments
input_dir = args.caps_dir
output_results = args.output_results
tsv_path = args.tsv_path
lr = args.lr
beta1 = args.beta1
beta2 = args.beta2
n_splits = args.n_splits
batch_size = args.batch_size
num_epoch = args.n_epoch
diagnoses = ['gaudo_1']
baseline = 'False'
split = None
mode = 'image'
preprocessing = 't1-linear'
num_workers = 2

# Parameters for Adam optimizer
#lr = 0.0001
#beta1 = 0.5
#beta2 = 0.999
# Parameters for split
#n_splits = 2
#split = None
# Parameters of input
#tsv_path = '/export/home/cse180022/apprimage_simo/local_image_processing/image_synthesis/output_results/tsv_files'
#diagnoses = ['gaudo_1'] ## to change ## diagnoses will be gado, not_gado
#input_dir = '/export/home/cse180022/apprimage_simo/image_preprocessing_data/ds9_caps'
#output_results = '/export/home/cse180022/apprimage_simo/local_image_processing/image_synthesis/output_results/cgan'
# Create dataloaders
#batch_size = 2
#baseline = 'False'
#num_epoch = 500
# Parameter for operations on dataloader
#mode = 'image'
#preprocessing = 't1-linear'
#num_workers = 2 #n_proc



transformations = get_transforms(mode, minmaxnormalization=True)


fold_iterator = range(n_splits)


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
    data_valid = MRIDatasetImage(
        input_dir,
        training_df,
        preprocessing,
        transformations=transformations)

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

    valid_loader = DataLoader(
        data_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )



    output_results_fold = os.path.join(output_results, 'fold-' + str(fi))
    if not os.path.exists(output_results_fold):
        os.makedirs(output_results_fold)
    # Train the generator
    generator = train_cgan(train_loader, valid_loader,output_results_fold, input_dir,
                       num_epoch,
                            lr=lr, beta1=beta1, beta2=beta2)
