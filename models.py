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
        ### pour residual: self.model(x) + x
    ##### au as


class Res(nn.Module):

    def __init__(self, in_size, out_size, n_conv):
        super(Res, self).__init__()
        if n_conv == 1:
            self.model = nn.Sequential(
            nn.Conv3d(in_size, out_size, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
            )
        elif n_conv == 2:
            self.model = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.Conv3d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2)
            )
        elif n_conv == 3:
            self.model = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.Conv3d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.Conv3d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        return self.model(x) + x



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

        self.down1 = UNetDown(in_channels, 128)
        self.down2 = UNetDown(128, 256)
        self.down3 = UNetDown(256, 512)
        self.down4 = UNetDown(512, 1024)
        self.down5 = UNetDown(1024, 1024)

        self.up1 = UNetUp(1024, 1024)
        self.up2 = UNetUp(2048, 512)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)

        self.final = FinalLayer(256, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        return self.final(u4, d1)

class GeneratorUNetResMod(nn.Module):
    """
    The generator will have a U-Net architecture with the following characteristics:

    the descending blocks are convolutional layers followed by instance normalization with a LeakyReLU activation function;

    the ascending blocks are transposed convolutional layers followed by instance normalization with a ReLU activation function.

    """
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNetResMod, self).__init__()

        self.r1 = Res(128, 128, 1) ##dr1
        self.r2 = Res(256, 256, 2) ##dr2
        self.r3 = Res(512, 512, 3) ##dr3
        self.r4 = Res(1024, 1024, 3) ##dr4, dr5

        self.down1 = UNetDown(in_channels, 128)
        self.down2 = UNetDown(128, 256)
        self.down3 = UNetDown(256, 512)
        self.down4 = UNetDown(512, 1024)
        self.down5 = UNetDown(1024, 1024)

        self.up1 = UNetUp(1024, 1024)
        self.up2 = UNetUp(2048, 512)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)

        self.final = FinalLayer(256, 1)

    def forward(self, x):

        d1 = self.down1(x)
        dr1 = self.r1(d1)

        d2 = self.down2(dr1)
        dr2 = self.r2(d2)

        d3 = self.down3(dr2)
        dr3 = self.r3(d3)

        d4 = self.down4(dr3)
        dr4 = self.r4(d4)

        d5 = self.down5(dr4) ## out= 1024
        dr5 = self.r4(d5) ## 1024, 2024

        u1 = self.up1(dr5)
        ur1 = self.r4(u1) ## 1024, 2024

        u2 = self.up2(ur1, d4) ## out= 512
        ur2 = self.r3(u2) ## 512, 512

        u3 = self.up3(ur2, d3) ## out=256
        ur3 = self.r2(u3)

        u4 = self.up4(ur3, d2) ## out = 128
        ur4 = self.r1(u4)

        return self.final(ur4, dr1)

################## Discriminator ###########################

class GeneratorUNetResMod_64(nn.Module):
    """
    The generator will have a U-Net architecture with the following characteristics:

    the descending blocks are convolutional layers followed by instance normalization with a LeakyReLU activation function;

    the ascending blocks are transposed convolutional layers followed by instance normalization with a ReLU activation function.

    """
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNetResMod_64, self).__init__()

        self.r1 = Res(64, 64, 1) ##dr1
        self.r2 = Res(128, 128, 2) ##dr2
        self.r3 = Res(256, 256, 3) ##dr3
        self.r4 = Res(512, 512, 3) ##dr4, dr5

        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = FinalLayer(128, 1)

    def forward(self, x):

        d1 = self.down1(x)
        dr1 = self.r1(d1)

        d2 = self.down2(dr1)
        dr2 = self.r2(d2)

        d3 = self.down3(dr2)
        dr3 = self.r3(d3)

        d4 = self.down4(dr3)
        dr4 = self.r4(d4)

        d5 = self.down5(dr4) ## out= 1024
        dr5 = self.r4(d5) ## 1024, 2024

        u1 = self.up1(dr5)
        ur1 = self.r4(u1) ## 1024, 2024

        u2 = self.up2(ur1, d4) ## out= 512
        ur2 = self.r3(u2) ## 512, 512

        u3 = self.up3(ur2, d3) ## out=256
        ur3 = self.r2(u3)

        u4 = self.up4(ur3, d2) ## out = 128
        ur4 = self.r1(u4)

        return self.final(ur4, dr1)


class GeneratorUNetResMod_Concat_Layer(nn.Module):
    """
    The generator will have a U-Net architecture with the following characteristics:

    the descending blocks are convolutional layers followed by instance normalization with a LeakyReLU activation function;

    the ascending blocks are transposed convolutional layers followed by instance normalization with a ReLU activation function.

    """
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNetResMod_Concat_Layer, self).__init__()

        self.r1 = Res(64, 64, 1) ##dr1
        self.r2 = Res(128, 128, 2) ##dr2
        self.r3 = Res(256, 256, 3) ##dr3
        self.r4 = Res(512, 512, 3) ##dr4, dr5

        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)

        self.up1 = UNetUp(64, 128)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = FinalLayer(128, 1)

    def forward(self, x):

        d1 = self.down1(x)
        dr1 = self.r1(d1)

        d2 = self.down2(dr1)
        dr2 = self.r2(d2)

        d3 = self.down3(dr2)
        dr3 = self.r3(d3)

        d4 = self.down4(dr3)
        dr4 = self.r4(d4)

        d5 = self.down5(dr4) ## out= 1024
        dr5 = self.r4(d5) ## 1024, 2024

        u1 = self.up1(dr5)
        ur1 = self.r4(u1) ## 1024, 2024

        u2 = self.up2(ur1, d4) ## out= 512
        ur2 = self.r3(u2) ## 512, 512

        u3 = self.up3(ur2, d3) ## out=256
        ur3 = self.r2(u3)

        u4 = self.up4(ur3, d2) ## out = 128
        ur4 = self.r1(u4)

        return self.final(ur4, dr1)
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
        layers.append(nn.Conv3d(512, 1, 4, padding=0))
        layers.append(nn.AvgPool3d(5))
        self.model = nn.Sequential(*layers)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class DiscriminatorCycle(nn.Module):
    def __init__(self, in_channels=1):
        super(DiscriminatorCycle, self).__init__()

        layers = []
        layers.extend(discriminator_block(in_channels, 64))
        layers.extend(discriminator_block(64, 128))
        layers.extend(discriminator_block(128, 256))
        layers.extend(discriminator_block(256, 512))
        layers.append(nn.Conv3d(512, 1, 4, padding=0))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

