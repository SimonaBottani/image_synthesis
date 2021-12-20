# image_synthesis
## Homogeneization of brain MRI from a clinical data warehouse using contrast-enhanced to non-contrast-enhanced image translation

Implementation of contrast-enhanced to non contrast-enhanced image translation of 3D T1w brain MRI

This repository contains the code for the work described in Bottani et al. 2021 (full reference below) for the image synthesis of non contrast-enhanced image translation of T1w brain MRI from contrast-enhanced MRI using 3D U-Net like models and conditional GAN


#### Dependencies
- To install all the python dependencies, please follow the instructionf of ` ClinicaDL` (https://github.com/aramis-lab/clinicadl)
- To preprocess the images, see ` clinica run t1-linear` and `clinica run deeplearning-prepare-data` from https://www.clinica.run)

#### Input necessary

Compulsory args:

- CAPS folder: It contains preprocessed images in the MNI space 
- OUTPUT folder: where results will be saved
- participants tsv: tsv path to the tsv containing participant / session id and diagnosis id for all the images. It is stored in a folder with the following path: `/path/to/tsv/fold-N` with `N` the index of the CV. The tsv file has the following structure to handle paired images:

| participant_id | session_id_1 | diagnosis_1 | session_id_2 | diagnosis_2 |
|----------------|:------------:|------------:|-------------:| -----------:|
| sub-0001       |  ses-M000    | gado_0      | ses-M001     |  gado_1     |
| sub-0002       |  ses-M110    | gado_0      | ses-M111     |  gado_1     |

- model: it can be ` generator` if only 3D U-Net or `conditional_gan` 
- generator_name: name of the `generator` model (for 3D Res-U-Net `GeneratorUNetResMod`, for 3D Att-U-Net `AttU_Net`, for 3D Trans-U-Net `TransUNet`). If you use `Trans-U-Net` please cite: `Wang et al, 2021` (see full reference below)

Optional args:

- n_epoch: number of epochs
- lr: learning rate
- n_splits
- batch_size
- input_dim: size of the image (i.e. 128 128 128)
- skull_strip: skull_strip if used skull_stripped images
- generator_pretrained: path to the pretrained generator if exists
- discriminator_pretrained: path to the pretrained discriminator if exists
- train_genetator: True if train generator 




#### How it works

- python main.py + compulsory args + optional args
- python main_test.py + compulsory args + optional args

#### Final output
Model able to obtain images non contrast-enhanced to contrast-enhanced 3D T1w brain MRI

## Citing this work

> Bottani, Simona, Elina Thibeau-Sutre, Aurélien Maire, Sebastian Ströer, Didier Dormont, Olivier Colliot, Ninon Burgos, and Apprimage Study Group. "Homogenization of brain MRI from a clinical data warehouse using contrast-enhanced to non-contrast-enhanced image translation with U-Net derived models." In SPIE-Medical Imaging. 2022. Available on Hal: https://hal.archives-ouvertes.fr/hal-03478798/

If you use the `Trans U-Net ` please cite:
> Wang W, Chen C, Ding M, Li J, Yu H, Zha S. TransBTS: Multimodal Brain Tumor Segmentation Using Transformer, International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2021 (https://github.com/Wenxuan-1119/TransBTS)
