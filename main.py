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
from models import GeneratorUNet, Discriminator, Discriminator64, DiscriminatorCycle, GeneratorUNetResMod, R2AttU_Net, AttU_Net, R2U_Net, BTS



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
    'model_names',
    help='Name of the type of the model used',
    default='generator', nargs='+', type=str,
    choices=['generator', 'conditional_gan', 'cycle_gan']
)

parser.add_argument(
    '--generator_name',
    help='Name of the type of the model used',
    default='GeneratorUNet', nargs='+', type=str,
    choices=['GeneratorUNet', 'GeneratorUNetResMod', 'AttU_Net',
             'R2AttU_Net', 'R2U_Net', 'TransUNet']
)

parser.add_argument(
    '--generator_pretrained',
    help='Path to the pretrained generator',
    default='.', nargs='+', type=str,
)

parser.add_argument(
    '--discriminator_pretrained',
    help='Path to the pretrained discriminator',
    default='.', nargs='+', type=str,
)

parser.add_argument(
        '--n_epoch',
        type=int,
        default=200,
        help='number of epoch'
    )

parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning_rate'
    )

parser.add_argument(
    '--beta1',
    type=float,
    default=0.5,
    help='beta1 for Adam Optimizer'
)

parser.add_argument(
    '--beta2',
    type=float,
    default=0.999,
    help='beta1 for Adam Optimizer'
)

parser.add_argument(
    '--input_dim',
    type=float,
    default=128,
    help='trilinear interpolation of input, i.e. 128/64'
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
    default=1,
    help='batch_size'
)

parser.add_argument(
    '--n_gpu',
    type=int,
    default=3,
    help='number_id_gpu'
)
parser.add_argument(
    '--train_generator',
    type=str,
    default='train_generator',
    help='train generator yes or not '
)

parser.add_argument(
    '--n_patches',
    type=int,
    default=1,
    help='number of patches'
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
input_dim = args.input_dim
generator_pretrained = args.generator_pretrained
discriminator_pretrained = args.discriminator_pretrained
n_patches = args.n_patches

train_gen = args.train_generator


if train_gen == 'train_generator':
    print('i am training the generator')
    train_gen = True
else:
    train_gen = False



transformations = get_transforms(mode, minmaxnormalization=True)

fold_iterator = range(n_splits)
if torch.cuda.is_available():
    print('>> GPU available.')
    DEVICE = torch.device('cuda')
    torch.cuda.set_device(n_gpu)


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
            transformations=transformations,
            skull_strip=skull_strip)
    data_valid = MRIDatasetImage(
            input_dir,
            valid_df,
            preprocessing,
            transformations=transformations,
            skull_strip=skull_strip
    )

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

    if model == ['generator']:
        if model_generator == ['GeneratorUNetResMod']:
            print('i am here')
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
            print('Trying Transformer')
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


        generator = train_generator(train_loader, valid_loader, output_results_fold, input_dir,
                                    model_generator,
                                   num_epoch,
                                   lr=lr, beta1=beta1, beta2=beta2, skull_strip=skull_strip,
                                    input_dim=128)

    elif model == ['conditional_gan']:
        if model_generator == ['GeneratorUNetResMod']:
            print(model_generator)
            model_generator = GeneratorUNetResMod()
        elif model_generator == ['GeneratorUNet']:
            print(model_generator)
            model_generator = GeneratorUNet()
        elif model_generator == ['AttU_Net']:
            print(model_generator)
            model_generator = AttU_Net()
        elif model_generator == ['TransUNet']:
            print(model_generator)
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
        ## if model == 'conditional_gan' load generator already trained
        print(generator_pretrained[0])

        param_dict = torch.load(os.path.join(generator_pretrained[0], 'fold-0/generator/best_loss',
                                             'model_best.pth.tar'), map_location="cpu")
        model_generator.load_state_dict(param_dict['model'])
        print('Model generator uploaded')

        print(discriminator_pretrained[0])

        if discriminator_pretrained != ['False']:
            print('Loading discriminator')
            model_discriminator = Discriminator64()
            param_dict = torch.load(os.path.join(discriminator_pretrained[0], 'fold-0/discriminator/best_loss',
                                             'model_best.pth.tar'), map_location="cpu")
            model_discriminator.load_state_dict(param_dict['model'])
            print('Model discriminator uploaded')
        elif discriminator_pretrained == ['False']:
            model_discriminator = Discriminator64()




        generator = train_cgan(train_loader, valid_loader,output_results_fold, input_dir,
                            model_generator, model_discriminator,
                               num_epoch,
                                lr=lr, beta1=beta1, beta2=beta2, skull_strip=skull_strip,
                               train_gen=train_gen)

    elif model == ['cycle_gan']:
        generator = train_cyclegan(train_loader, valid_loader,output_results_fold, input_dir,
                           num_epoch,
                                lr=lr, beta1=beta1, beta2=beta2)




    evaluate_generator(generator, train_loader, output_results_fold, modality='train')
    evaluate_generator(generator, valid_loader, output_results_fold, modality='valid')


