
# torch stuff
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

import pandas as pd
import os
from metrics import *

def evaluate_generator(generator, batch_loader, output_results_fold, modality='train'):
    """Evaluate a generator.

    Args:
        generator: (GeneratorUNet) neural network generating T2-w images
        batch_loader: train_loader, valid_loader or test_loader
        modality: 'train', 'valid', 'test'

    """
    res = []

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for i, batch in enumerate(batch_loader):

        # Inputs T1-w and T2-w
        real_t1 = Variable(batch["T1"].type(Tensor), requires_grad=False)
        real_t2 = Variable(batch["T2"].type(Tensor), requires_grad=False)

        real_1 = F.interpolate(real_1, size=(64, 64, 64), mode='trilinear', align_corners=False)
        real_2 = F.interpolate(real_2, size=(64, 64, 64), mode='trilinear', align_corners=False)
        real_1[real_1 != real_1] = 0
        real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
        real_2[real_2 != real_2] = 0
        real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())

        fake_t2 = Variable(generator(real_t1), requires_grad=False)

        mae = mean_absolute_error(real_t2, fake_t2).item()
        psnr = peak_signal_to_noise_ratio(real_t2, fake_t2).item()
        ssim = structural_similarity_index(real_t2, fake_t2).item()

        res.append([mae, psnr, ssim])


    df = pd.DataFrame([
        pd.DataFrame(res, columns=['MAE', 'PSNR', 'SSIM']).mean().squeeze()
    ], index=[modality]).T

    df.to_csv(os.path.join(output_results_fold, 'metric_evaluation_' + modality + '.tsv'), sep='\t')


    return df