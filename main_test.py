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

parser.add_argument(
    '--generator_name',
    help='Name of the type of the model used',
    default='GeneratorUNet', nargs='+', type=str,
    choices=['GeneratorUNet', 'GeneratorUNetResMod', 'AttU_Net', 'TransUNet']
)
parser.add_argument(
    '--input_dim',
    type=float,
    default=128,
    help='trilinear interpolation of input, i.e. 128/64'
)
parser.add_argument(
    '--n_gpu',
    type=int,
    default=3,
    help='number_id_gpu'
)

parser.add_argument(
    '--real_im_exists',
    type=int,
    default=1,
    help='if real_im to compare exists: 1, if not : 0'
)

parser.add_argument(
    '--name_test_folder',
    type=str,
    default='test_images',
    help='name of the test folder'
)



args = parser.parse_args()

## write command line arguments on json
commandline = parser.parse_known_args()
commandline_to_json(commandline, test=True)

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
n_gpu = args.n_gpu
model_generator = args.generator_name
real_im_exists = args.real_im_exists
name_test_folder = args.name_test_folder
input_dim = args.input_dim




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
        torch.cuda.set_device(n_gpu)

    if model == ['generator']:
        if model_generator == ['GeneratorUNetResMod']:
            model_generator = GeneratorUNetResMod()
        elif model_generator == ['GeneratorUNet']:
            model_generator = GeneratorUNet()
        elif model_generator == ['R2U_Net']:
            model_generator = R2U_Net()
        elif model_generator == ['AttU_Net']:
            model_generator = AttU_Net()
        elif model_generator == ['R2AttU_Net']:
            model_generator = R2AttU_Net()
        elif model_generator == ['TransUNet']:
            model_generator = BTS(img_dim=128,
                                  patch_dim=8,
                                  num_channels=1,
                                  num_classes=1,
                                  embedding_dim=512,
                                  num_heads=8,
                                  num_layers=4,
                                  hidden_dim=4096,
                                  dropout_rate=0.1,
                                  attn_dropout_rate=0.1,
                                  conv_patch_representation=True,
                                  positional_encoding_type="learned",
                                  )

        generator = model_generator
        if cuda:
            generator = generator.cuda()


        param_dict = torch.load(os.path.join(output_results_fold, 'generator/best_loss',
                                             'model_best.pth.tar'), map_location="cpu")
        generator.load_state_dict(param_dict['model'])

        evaluate_generator(generator, test_loader, output_results_fold, modality='test',
                           input_dim=128)

    ### save images
    print('save files')
    if not os.path.exists(os.path.join(output_results_fold, name_test_folder)):
        os.makedirs(os.path.join(output_results_fold, name_test_folder))

    sample_images_testing(generator, test_loader, input_dir, os.path.join(output_results_fold, name_test_folder),
                          skull_strip)

