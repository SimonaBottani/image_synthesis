
# torch stuff
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import scipy
import pandas as pd
import os
from metrics import *
import numpy as np
import copy

def evaluate_generator(generator, batch_loader, output_results_fold, modality='train',
                       input_dim=128):
    """Evaluate a generator.

    Args:
        generator: (GeneratorUNet) neural network generating T2-w images
        batch_loader: train_loader, valid_loader or test_loader
        modality: 'train', 'valid', 'test'

    """
    res_mae = []
    res_pnsr = []
    res_ssim = []
    res_mae_12 = []
    res_pnsr_12 = []
    res_ssim_12 = []
    participant_id = []
    session_id = []

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for i, batch in enumerate(batch_loader):

        # Inputs T1-w and T2-w
        real_1 = Variable(batch["image_1"].type(Tensor), requires_grad=False)
        real_2 = Variable(batch["image_2"].type(Tensor), requires_grad=False)

        ####### READ the Skull-stripped iamges to create the mask - in 169-208-179
        real_1_sk_st = Variable(batch["image_path_1_skull_strip"].type(Tensor), requires_grad=False)
        real_2_sk_st = Variable(batch["image_path_2_skull_strip"].type(Tensor), requires_grad=False)
        #real_1_sk_st = F.interpolate(real_1_sk_st, size=(128, 128, 128), mode='trilinear', align_corners=False)
        #real_2_sk_st = F.interpolate(real_2_sk_st, size=(128, 128, 128), mode='trilinear', align_corners=False)
        real_1_sk_st[real_1_sk_st != real_1_sk_st] = 0
        real_1_sk_st = (real_1_sk_st - real_1_sk_st.min()) / (real_1_sk_st.max() - real_1_sk_st.min())
        real_2_sk_st[real_2_sk_st != real_2_sk_st] = 0
        real_2_sk_st = (real_2_sk_st - real_2_sk_st.min()) / (real_2_sk_st.max() - real_2_sk_st.min())
        real_1_sk_st = real_1_sk_st[0, 0, :, :, :]
        real_2_sk_st = real_2_sk_st[0, 0, :, :, :]

        real_1 = F.interpolate(real_1, size=(input_dim, input_dim, input_dim), mode='trilinear', align_corners=False)
        real_2 = F.interpolate(real_2, size=(input_dim, input_dim, input_dim), mode='trilinear', align_corners=False)
        real_1[real_1 != real_1] = 0
        real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
        real_2[real_2 != real_2] = 0
        real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())

        fake_2 = Variable(generator(real_1), requires_grad=False)

        print('testing on 169, 208, 179')
        real_1 = F.interpolate(real_1, size=(169, 208, 179), mode='trilinear', align_corners=False)
        real_2 = F.interpolate(real_2, size=(169, 208, 179), mode='trilinear', align_corners=False)
        fake_2 = F.interpolate(fake_2, size=(169, 208, 179), mode='trilinear', align_corners=False)

        ## create mask for the metrics

        ### first method:
        real_1 = real_1[0, 0, :, :, :]
        real_2 = real_2[0, 0, :, :, :]
        fake_2 = fake_2[0, 0, :, :, :]

        real_1_mask = copy.deepcopy(real_1_sk_st) ### deepcopy !!!
        real_2_mask = copy.deepcopy(real_2_sk_st) ### deepcopy !!!

        real_1_mask[real_1_mask != 0] = 1
        real_2_mask[real_2_mask != 0] = 1

        mask = real_1_mask + real_2_mask
        mask[mask != 0] = 1
        #c_mask = scipy.ndimage.morphology.binary_fill_holes(c, structure=None, output=None, origin=0)
        #c_mask = nib.Nifti1Image(c_mask, affine=affine, header=header)
        fake_2_masked = copy.deepcopy(fake_2) ### deepcopy !!!
        real_2_masked = copy.deepcopy(real_2) ### deepcopy !!!
        real_2_masked[mask == 0] = 0
        fake_2_masked[mask == 0] = 0

        real_1_masked = copy.deepcopy(real_1)
        real_1_masked[mask == 0] = 0

        mae = mean_absolute_error(real_2_masked, fake_2_masked).item()
        psnr = peak_signal_to_noise_ratio(real_2_masked, fake_2_masked).item()
        ssim = structural_similarity_index(real_2_masked, fake_2_masked).item()
        res_mae.append(mae)
        res_pnsr.append(psnr)
        res_ssim.append(ssim)
        participant_id.append(batch['participant_id'][0]) ###batch size must be 1 !!
        session_id.append(batch['session_id_2'][0])

        mae = mean_absolute_error(real_2_masked, real_1_masked).item()
        psnr = peak_signal_to_noise_ratio(real_2_masked, real_1_masked).item()
        ssim = structural_similarity_index(real_2_masked, real_1_masked).item()
        res_mae_12.append(mae)
        res_pnsr_12.append(psnr)
        res_ssim_12.append(ssim)

        #res.append([mae, psnr, ssim])


    #df = pd.DataFrame([
    #    pd.DataFrame(res, columns=['MAE', 'PSNR', 'SSIM']).mean().squeeze()
    #], index=[modality]).T
    df = pd.DataFrame(data = {'MAE':res_mae, 'PSNR':res_pnsr, 'SSIM': res_ssim,
                              'MAE_12': res_mae_12, 'PSNR_12': res_pnsr_12, 'SSIM_12': res_ssim_12,
                             'participant_id': participant_id, 'session_id': session_id})


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


def write_validation_tsv(epoch, valid_loader, output_results, generator, criterion,
                         input_dim):
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
        real_1 = F.interpolate(real_1, size=(input_dim, input_dim, input_dim), mode='trilinear', align_corners=False)
        real_2 = F.interpolate(real_2, size=(input_dim, input_dim, input_dim), mode='trilinear', align_corners=False)

        real_1[real_1 != real_1] = 0
        real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
        real_2[real_2 != real_2] = 0
        real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())
        with torch.no_grad():
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

