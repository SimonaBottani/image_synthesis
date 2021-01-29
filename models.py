# torch stuff
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

# torchsummary and torchvision
from torchsummary import summary
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



################## Generator ###########################

class UNetDown(nn.Module):
    """Descending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size):
        super(UNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_size, out_size, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2)
          )

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Ascending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_size, out_size, kernel_size=4,
                               stride=2, padding=1),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


class FinalLayer(nn.Module):
    """Final block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size):
        super(FinalLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


class GeneratorUNet(nn.Module):
    """
    The generator will have a U-Net architecture with the following characteristics:

    the descending blocks are convolutional layers followed by instance normalization with a LeakyReLU activation function;

    the ascending blocks are transposed convolutional layers followed by instance normalization with a ReLU activation function.

    """
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(in_channels, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = FinalLayer(128, 1)

    def forward(self, x):
        x = F.interpolate(x, size=(64, 64, 64), mode='trilinear', align_corners=False)
        print(x.shape)
        print('down')
        d1 = self.down1(x)
        print(d1.shape)
        d2 = self.down2(d1)
        print(d2.shape)
        d3 = self.down3(x)
        print(d3.shape)
        d4 = self.down4(d3)
        print(d4.shape)
        d5 = self.down5(d4)
        print(d5.shape)
        print('up')

        u1 = self.up1(d5)
        print(u1.shape)
        print('input for skip connection U2')
        print(u1.shape)
        print(d4.shape)
        u2 = self.up2(u1, d4)
        print('input for skip connection U3')
        print(u2.shape)
        print(d3.shape)
        u3 = self.up3(u2, d3)
        print('input for skip connection U4')
        print(u3.shape)
        print(d2.shape)
        u4 = self.up4(u3, d2)

        return self.final(u4, d1)


################## Discriminator ###########################


def discriminator_block(in_filters, out_filters):
    """Return downsampling layers of each discriminator block"""
    layers = [nn.Conv3d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        layers = []
        layers.extend(discriminator_block(in_channels * 2, 64))
        layers.extend(discriminator_block(64, 128))
        layers.extend(discriminator_block(128, 256))
        layers.extend(discriminator_block(256, 512))
        layers.append(nn.Conv2d(512, 1, 4, padding=0))

        self.model = nn.Sequential(*layers)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

