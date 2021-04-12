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
from train import train_cgan, train_generator, train_cyclegan
from utils import *
from evaluation import *
from models import *


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
        'tsv_test',
        help='Test tsv',
        default=None
    )

parser.add_argument(
    'model_names',
    help='Name of the type of the model used',
    default='generator', nargs='+', type=str,
    choices=['generator', 'conditional_gan', 'cycle_gan']
)

parser.add_argument(
    '--skull_strip',
    type=str,
    default='skull_strip',
    help='skull strip if skull_strip '
)


parser.add_argument(
    '--n_splits',
    type=int,
    help='number of CV'
)


parser.add_argument(
    '--batch_size',
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
tsv_path = args.tsv_test
n_splits = args.n_splits
batch_size = args.batch_size
model = args.model_names
diagnoses = ['gaudo_1']
baseline = 'False'
split = None
mode = 'image'
preprocessing = 't1-linear'
num_workers = 2
skull_strip = args.skull_strip




transformations = get_transforms(mode, minmaxnormalization=True)

fold_iterator = range(n_splits)


for fi in fold_iterator:

    testing_df = pd.read_csv(tsv_path, sep='\t')

    data_test = MRIDatasetImage(
        input_dir,
        testing_df,
        preprocessing,
        transformations=transformations,
    skull_strip=skull_strip)


    test_loader = DataLoader(
            data_test,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )


    output_results_fold = os.path.join(output_results, 'fold-' + str(fi))
    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used
    if cuda == True:


        DEVICE = torch.device('cuda')
        torch.cuda.set_device(3)

    if model == ['generator']:
        generator = GeneratorUNet()
        if cuda:
            generator = generator.cuda()


        param_dict = torch.load(os.path.join(output_results_fold, 'generator/best_loss',
                                             'model_best.pth.tar'), map_location="cpu")
        generator.load_state_dict(param_dict['model'])


        if cuda:
            generator = generator.cuda()

    evaluate_generator(generator, test_loader, output_results_fold, modality='test')
    ### save images
    print('save files')
    if not os.path.exists(os.path.join(output_results_fold, 'test_images')):
        os.makedirs(os.path.join(output_results_fold, 'test_images'))
    sample_images_testing(generator, test_loader, input_dir, os.path.join(output_results_fold, 'test_images'),
                          skull_strip)

