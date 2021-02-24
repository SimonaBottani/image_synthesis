import torch.nn.functional as F
import os
import nibabel as nib
import torch


participant_id = ['sub-A6690502920504586872']
session_id = ['ses-M000']
i = 0
real_image_path = (os.path.join('/export/home/cse180022/apprimage_simo/image_preprocessing_data/ds9_caps',
                      'subjects', participant_id[i], 'ses-M000',
                                 't1_linear',participant_id[i] + '_' + 'ses-M000' + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz'))

real_image = nib.load(real_image_path).get_data()
header = nib.load(real_image_path).header
affine = nib.load(real_image_path).affine

### I need to rescale this
real_image = torch.from_numpy(real_image)
real_image = real_image.unsqueeze(0)
real_image = real_image.unsqueeze(0)
real_image = F.interpolate(real_image, size=(128, 128, 128), mode='trilinear', align_corners=False)
real_image[real_image != real_image] = 0 #replace nan by 0
real_image = (real_image - real_image.min()) / (real_image.max() - real_image.min()) #replace by 0
real_image = real_image.detach().cpu().numpy()
real_image = nib.Nifti1Image(real_image[0,0,:,:,:], affine=affine, header=header)

real_image.to_filename(os.path.join(
    '/export/home/cse180022/apprimage_simo/local_image_processing/image_synthesis/out_128/generator/eal_image_rescaled.nii.gz'))


#gen_image = nib.load(os.path.join('/export/home/cse180022/apprimage_simo/image_synthesis/out_128',
#                                  'epoch-' + str(epoch) + '_' + participant_id[i]
#                                  + participant_id[0] + '_reconstructed.nii.gz'))

