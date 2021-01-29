# torch stuff
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

# torchsummary and torchvision
#from torchsummary import summary
from torchvision.utils import save_image

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
from models import GeneratorUNet, Discriminator


def train_cgan(train_loader, test_loader, output_results,
               num_epoch=500,
               lr=0.0001, beta1=0.9, beta2=0.999):
    """Train a conditional GAN.

    Args:
        train_loader: (DataLoader) a DataLoader wrapping a the training dataset
        test_loader: (DataLoader) a DataLoader wrapping a the test dataset
        num_epoch: (int) number of epochs performed during training
        lr: (float) learning rate of the discriminator and generator Adam optimizers
        beta1: (float) beta1 coefficient of the discriminator and generator Adam optimizers
        beta2: (float) beta1 coefficient of the discriminator and generator Adam optimizers

    Returns:
        generator: (nn.Module) the trained generator
    """

    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Output folder
    if not os.path.exists(os.path.join(output_results, 'cgan')):
        os.makedirs(os.path.join(output_results, 'cgan'))

    # Loss functions
    criterion_GAN = torch.nn.BCEWithLogitsLoss()   # To complete. A loss adapted to binary classification like torch.nn.BCEWithLogitsLoss
    criterion_pixelwise = torch.nn.L1Loss()    # To complete. A loss for a voxel-wise comparison of images like torch.nn.L1Loss

    lambda_GAN = 1.  # Weights criterion_GAN in the generator loss
    lambda_pixel = 1.  # Weights criterion_pixelwise in the generator loss

    # Initialize generator and discriminator
    generator = GeneratorUNet()   # To complete
    discriminator = Discriminator() # To complete

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    # Optimizers
    optimizer_generator = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    def sample_images(epoch):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(test_loader))
        real_1 = Variable(imgs["image_1"].type(Tensor))
        real_2 = Variable(imgs["image_2"].type(Tensor))
        fake_2 = generator(real_1)
        img_sample = torch.cat((real_1.data, fake_2.data, real_2.data), -2)
        save_image(img_sample, os.path.join(output_results, 'cgan/epoch-' + epoch + ".nii.gz"),
                   nrow=5, normalize=True)

    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    for epoch in range(num_epoch):
        for i, data in enumerate(train_loader, 0):

            # Inputs T1-w and T2-w
            real_1 = data["image_1"].type(Tensor)
            real_2 = data["image_2"].type(Tensor)

            real_1[real_1 != real_1] = 0
            real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
            real_2[real_2 != real_2] = 0
            real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())


            # Create labels
            valid = Variable(Tensor(np.ones((real_2.size(0), 1, 1, 1))),
                             requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_2.size(0), 1, 1, 1))),
                            requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_generator.zero_grad()

            # GAN loss
            fake_2 =  generator(real_1)  # To complete
            pred_fake = discriminator(fake_2, real_1)
            loss_GAN = criterion_GAN(pred_fake, valid)

            # L1 loss
            loss_pixel = criterion_pixelwise(fake_2, real_2)

            # Total loss
            loss_generator = lambda_GAN * loss_GAN + lambda_pixel * loss_pixel

            # Compute the gradient and perform one optimization step
            loss_generator.backward()
            optimizer_generator.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_discriminator.zero_grad()

            # Real loss
            pred_real = discriminator(real_2, real_1)   # To complete
            loss_real =  criterion_GAN(pred_real, valid)  # To complete

            # Fake loss
            pred_fake = discriminator(fake_2.detach(), real_1)   # To complete
            loss_fake = criterion_GAN(pred_fake, fake)   # To complete

            # Total loss
            loss_discriminator = 0.5 * (loss_real + loss_fake)

            # Compute the gradient and perform one optimization step
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(train_loader) + i
            batches_left = num_epoch * len(train_loader) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] "
                "[G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch + 1,
                    num_epoch,
                    i,
                    len(train_loader),
                    loss_discriminator.item(),
                    loss_generator.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )

        # Save images at the end of each epoch
        sample_images(epoch)

    return generator