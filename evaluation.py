
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
        real_1 = Variable(batch["image_1"].type(Tensor), requires_grad=False)
        real_2 = Variable(batch["image_2"].type(Tensor), requires_grad=False)

        real_1 = F.interpolate(real_1, size=(128, 128, 128), mode='trilinear', align_corners=False)
        real_2 = F.interpolate(real_2, size=(128, 128, 128), mode='trilinear', align_corners=False)
        real_1[real_1 != real_1] = 0
        real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
        real_2[real_2 != real_2] = 0
        real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())

        fake_2 = Variable(generator(real_1), requires_grad=False)

        mae = mean_absolute_error(real_2, fake_2).item()
        psnr = peak_signal_to_noise_ratio(real_2, fake_2).item()
        ssim = structural_similarity_index(real_2, fake_2).item()

        res.append([mae, psnr, ssim])


    df = pd.DataFrame([
        pd.DataFrame(res, columns=['MAE', 'PSNR', 'SSIM']).mean().squeeze()
    ], index=[modality]).T

    df.to_csv(os.path.join(output_results_fold, 'metric_evaluation_' + modality + '.tsv'), sep='\t')


    return df

def sample_images_testing(generator, test_loader, caps_dir, output_results, skull_strip):
    """Saves a generated sample from the validation set"""

    import nibabel as nib

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for i, batch in enumerate(test_loader):


        real_1 = Variable(batch["image_1"].type(Tensor))
        real_2 = Variable(batch["image_2"].type(Tensor))

        ### reshape image
        real_1 = F.interpolate(real_1, size=(128, 128, 128), mode='trilinear', align_corners=False)
        real_2 = F.interpolate(real_2, size=(128, 128, 128), mode='trilinear', align_corners=False)

        real_1[real_1 != real_1] = 0
        real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
        real_2[real_2 != real_2] = 0
        real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())

        fake_2 = generator(real_1)

        if skull_strip != 'skull_strip':

            img_nifti = os.path.join(caps_dir, 'subjects', batch['participant_id'][0], batch['session_id_2'][0],
                                 't1_linear',
                                 batch['participant_id'][0] + '_' + batch['session_id_2'][0] +
                                 '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz')

        elif skull_strip == 'skull_strip':
            img_nifti = os.path.join(caps_dir, 'subjects', batch['participant_id'][0], batch['session_id_2'][0],
                                 't1_linear',
                                batch['participant_id'][0] + '_' + batch['session_id_2'][0] +
                                '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_skull-skripped-hdbet_T1w.nii.gz')

        header = nib.load(img_nifti).header
        affine = nib.load(img_nifti).affine
        fake_2 = fake_2.detach().cpu().numpy()
        fake_2_example = nib.Nifti1Image(fake_2[0,0,:,:,:], affine=affine, header=header)
        #if not os.path.exists(os.path.join(output_results, 'epoch-' + str(epoch))):
        #    os.makedirs(os.path.join(output_results, 'epoch-' + str(epoch)))
        fake_2_example.to_filename(os.path.join(output_results,
            batch['participant_id'][0] + '_' + batch['session_id_2'][0] + '_reconstructed.nii.gz'))


def write_validation_tsv(epoch, valid_loader, output_results, generator, criterion):
    """

    :param epoch: epoch
    :param valid_loader: validation loader
    :param output_results: where to save the file
    :param generator: generator UNet
    :param criterion: L1 loss
    :return: file for validation tsv
    """

    filename = os.path.join(output_results, 'validation.tsv')
    columns = ['epoch', 'batch', 'loss']

    # Tensor type (put everything on GPU if possibile)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for i, batch in enumerate(valid_loader):

        real_1 = Variable(batch["image_1"].type(Tensor))
        real_2 = Variable(batch["image_2"].type(Tensor))

        ### reshape image
        real_1 = F.interpolate(real_1, size=(128, 128, 128), mode='trilinear', align_corners=False)
        real_2 = F.interpolate(real_2, size=(128, 128, 128), mode='trilinear', align_corners=False)

        real_1[real_1 != real_1] = 0
        real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
        real_2[real_2 != real_2] = 0
        real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())

        fake_2 = generator(real_1)
        loss = criterion(fake_2, real_2)


    row = np.array(
        [epoch + 1, i,
         loss.item()]
    ).reshape(1, -1)
    row_df = pd.DataFrame(row, columns=columns)
    with open(filename, 'a') as f:
        row_df.to_csv(f, header=True, index=False, sep='\t')

    return loss

