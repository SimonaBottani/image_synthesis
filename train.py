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
from models import GeneratorUNet, Discriminator, DiscriminatorCycle, GeneratorUNetResMod
from evaluation import write_validation_tsv
from utils import save_checkpoint
# Nibabel
import nibabel as nib



def train_cgan(train_loader, test_loader, output_results,
                caps_dir, model_generator, discriminator,
               num_epoch=500,
               lr=0.0001, beta1=0.9, beta2=0.999, skull_strip=None,
               train_gen=True):
    """Train a conditional GAN.

    Args:
        train_loader: (DataLoader) a DataLoader wrapping a the training dataset
        test_loader: (DataLoader) a DataLoader wrapping a the test dataset
        num_epoch: (int) number of epochs performed during training
        lr: (float) learning rate of the discriminator and generator Adam optimizers
        lr: (float) learning rate of the discriminator and generator Adam optimizers
        beta1: (float) beta1 coefficient of the discriminator and generator Adam optimizers
        beta2: (float) beta1 coefficient of the discriminator and generator Adam optimizers

    Returns:
        generator: (nn.Module) the trained generator
    """
    best_valid_loss = np.inf
    columns = ['epoch', 'batch', 'loss_discriminator', 'loss_generator', 'loss_pixel', 'loss_GAN']
    filename = os.path.join(output_results, 'training.tsv')

    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    model_dir_generator = os.path.join(output_results, 'generator')
    model_dir_discriminator = os.path.join(output_results, 'discriminator')

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
    generator = model_generator


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

        ### reshape image
        real_1 = F.interpolate(real_1, size=(128, 128, 128), mode='trilinear', align_corners=False)
        real_2 = F.interpolate(real_2, size=(128, 128, 128), mode='trilinear', align_corners=False)

        real_1[real_1 != real_1] = 0
        real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
        real_2[real_2 != real_2] = 0
        real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())

        fake_2 = generator(real_1)

        if skull_strip != 'skull_strip':

            img_nifti = os.path.join(caps_dir, 'subjects', imgs['participant_id'][0], imgs['session_id_2'][0],
                                 't1_linear',
                                 imgs['participant_id'][0] + '_' + imgs['session_id_2'][0] +
                                 '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz')
        elif skull_strip == 'skull_strip':
            img_nifti = os.path.join(caps_dir, 'subjects', imgs['participant_id'][0], imgs['session_id_2'][0],
                                 't1_linear',
                                imgs['participant_id'][0] + '_' + imgs['session_id_2'][0] +
                                '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_skull-skripped-hdbet_T1w.nii.gz')

        header = nib.load(img_nifti).header
        affine = nib.load(img_nifti).affine
        fake_2 = fake_2.detach().cpu().numpy()
        fake_2_example = nib.Nifti1Image(fake_2[0,0,:,:,:], affine=affine, header=header)
        if not os.path.exists(os.path.join(output_results, 'validation_images' )):
            os.makedirs(os.path.join(output_results, 'validation_images'))

        fake_2_example.to_filename(os.path.join(output_results, 'validation_images', 'epoch-' + str(epoch) + '_' +
            imgs['participant_id'][0] + '_' + imgs['session_id_2'][0] + '_reconstructed.nii.gz'))



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

            ### Reshape input ###
            real_1 = F.interpolate(real_1, size=(128, 128, 128), mode='trilinear', align_corners=False)
            real_2 = F.interpolate(real_2, size=(128, 128, 128), mode='trilinear', align_corners=False)


            # Create labels
            #valid = Variable(Tensor(np.ones((real_2.size(0), 1, 1, 1, 1))),
            #                 requires_grad=False)
            ##soft label
            n = (0.3) * torch.rand((real_2.size(0), 1, 1, 1, 1))
            n = torch.Tensor(n).cuda()
            valid = Variable(Tensor(n),
                             requires_grad=False)
            n = (1 - 0.7) * torch.rand((real_2.size(0), 1, 1, 1, 1)) + 0.7
            n = torch.Tensor(n).cuda()
            fake = Variable(Tensor(n),
                             requires_grad=False)

            #fake = Variable(Tensor(np.zeros((real_2.size(0), 1, 1, 1, 1))),
            #                requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------

            if train_gen == True:
                print('generator_update')



                # GAN loss
                fake_2 = generator(real_1)  # To complete
                pred_fake = discriminator(fake_2, real_1)

                loss_GAN = criterion_GAN(pred_fake, valid) ## change with fake

                # L1 loss
                loss_pixel = criterion_pixelwise(fake_2, real_2)

                # Total loss
                loss_generator = lambda_GAN * loss_GAN + lambda_pixel * loss_pixel

                # Compute the gradient and perform one optimization step
                if i%2 == 0:
                    loss_generator.backward()
                    optimizer_generator.step()
                    optimizer_generator.zero_grad()

            # ---------------------
            #  Train Discriminator
            # ---------------------


            # Real loss
            pred_real = discriminator(real_2, real_1)   # To complete
            loss_real = criterion_GAN(pred_real, valid)  # To complete

            # Fake loss
            fake_2 = generator(real_1)
            pred_fake = discriminator(fake_2.detach(), real_1)   # To complete
            loss_fake = criterion_GAN(pred_fake, fake)   # To complete

            # Total loss
            loss_discriminator = 0.5 * (loss_real + loss_fake)

            # Compute the gradient and perform one optimization step
            if i%2 == 0:
                loss_discriminator.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()

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
            if train_gen == True:
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
            else:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] "
                    " ETA: %s"
                    % (
                        epoch + 1,
                        num_epoch,
                        i,
                        len(train_loader),
                        loss_discriminator.item(),
                        time_left,
                    )
                )

        # Save images at the end of each epoch
        if train_gen == True:
            columns = ['epoch', 'batch', 'loss_discriminator', 'loss_generator', 'loss_pixel', 'loss_GAN']
            row = np.array(
            [epoch + 1, i, loss_discriminator.item(), loss_generator.item(),
            loss_pixel.item(),
             loss_GAN.item()]
            ).reshape(1, -1)
        else:
            columns = ['epoch', 'batch', 'loss_discriminator']
            row = np.array(
            [epoch + 1, i, loss_discriminator.item()]
            ).reshape(1, -1)

        row_df = pd.DataFrame(row, columns=columns)
        with open(filename, 'a') as f:
            row_df.to_csv(f, header=True, index=False, sep='\t')

        if epoch % 20 == 0:
            sample_images(epoch)

        ###### save generator

        if train_gen == True:

            loss_valid = write_validation_tsv(epoch, train_loader, output_results, generator, criterion_pixelwise,
                                              128)

            loss_is_best = loss_valid < best_valid_loss
            best_valid_loss = min(loss_valid, best_valid_loss)

            save_checkpoint({'model': generator.state_dict(),
                             'epoch': epoch,
                             'valid_loss': loss_valid},
                            loss_is_best,
                            model_dir_generator)
            # Save optimizer state_dict to be able to reload
            save_checkpoint({'optimizer': optimizer_generator.state_dict(),
                             'epoch': epoch,
                             'name': loss_valid,
                             },
                            False,
                            model_dir_generator,
                            filename='optimizer.pth.tar')

            optimizer_generator.zero_grad()

        elif train_gen == False:
            loss_valid = write_validation_tsv(epoch, train_loader, output_results, generator, criterion_GAN,
                                              128)
            loss_is_best = loss_valid < best_valid_loss
            best_valid_loss = min(loss_valid, best_valid_loss)

        save_checkpoint({'model': discriminator.state_dict(),
                         'epoch': epoch,
                         'valid_loss': loss_valid},
                        loss_is_best,
                        model_dir_discriminator)
        # Save optimizer state_dict to be able to reload
        save_checkpoint({'optimizer': optimizer_discriminator.state_dict(),
                         'epoch': epoch,
                         'name': loss_valid,
                         },
                        False,
                        model_dir_discriminator,
                        filename='optimizer.pth.tar')
        del loss_valid
        optimizer_discriminator.zero_grad()

    return generator



def train_generator(train_loader, test_loader, output_results,
                    caps_dir, model_generator,
                    num_epoch=500,
                    lr=0.0001, beta1=0.9, beta2=0.999, skull_strip=None,
                    input_dim=128):
    """Train a generator on its own.

    Args:
        train_loader: (DataLoader) a DataLoader wrapping the training dataset
        test_loader: (DataLoader) a DataLoader wrapping the test dataset
        num_epoch: (int) number of epochs performed during training
        lr: (float) learning rate of the discriminator and generator Adam optimizers
        beta1: (float) beta1 coefficient of the discriminator and generator Adam optimizers
        beta2: (float) beta1 coefficient of the discriminator and generator Adam optimizers

    Returns:
        generator: (nn.Module) the trained generator
    """
    best_valid_loss = np.inf

    columns = ['epoch', 'batch', 'loss']
    filename = os.path.join(output_results, 'training.tsv')
    model_dir = os.path.join(output_results, 'generator')

    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Output folder
    if not os.path.exists(os.path.join(output_results, 'generator')):
        os.makedirs(os.path.join(output_results, 'generator'))

    # Loss function
    criterion = torch.nn.L1Loss()   # To complete. A loss for a voxel-wise comparison of images like torch.nn.L1Loss
    #criterion = torch.nn.MSELoss()

    # Initialize the generator
    generator = model_generator #GeneratorUNet()  # To complete.

    if cuda:
        generator = generator.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(generator.parameters(),
                                 lr=lr, betas=(beta1, beta2))

    def sample_images(epoch):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(test_loader))
        real_1 = Variable(imgs["image_1"].type(Tensor))
        real_2 = Variable(imgs["image_2"].type(Tensor))

        ### reshape image
        real_1 = F.interpolate(real_1, size=(input_dim, input_dim, input_dim), mode='trilinear', align_corners=False)
        real_2 = F.interpolate(real_2, size=(input_dim, input_dim, input_dim), mode='trilinear', align_corners=False)

        real_1[real_1 != real_1] = 0
        real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
        real_2[real_2 != real_2] = 0
        real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())

        fake_2 = generator(real_1)

        if skull_strip != 'skull_strip':

            img_nifti = os.path.join(caps_dir, 'subjects', imgs['participant_id'][0], imgs['session_id_2'][0],
                                 't1_linear',
                                 imgs['participant_id'][0] + '_' + imgs['session_id_2'][0] +
                                 '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz')
        elif skull_strip == 'skull_strip':
            img_nifti = os.path.join(caps_dir, 'subjects', imgs['participant_id'][0], imgs['session_id_2'][0],
                                 't1_linear',
                                imgs['participant_id'][0] + '_' + imgs['session_id_2'][0] +
                                '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_skull-skripped-hdbet_T1w.nii.gz')

        header = nib.load(img_nifti).header
        affine = nib.load(img_nifti).affine
        fake_2 = fake_2.detach().cpu().numpy()
        fake_2_example = nib.Nifti1Image(fake_2[0,0,:,:,:], affine=affine, header=header)
        if not os.path.exists(os.path.join(output_results, 'validation_images' )):
            os.makedirs(os.path.join(output_results, 'validation_images'))

        fake_2_example.to_filename(os.path.join(output_results, 'validation_images', 'epoch-' + str(epoch) + '_' +
            imgs['participant_id'][0] + '_' + imgs['session_id_2'][0] + '_reconstructed.nii.gz'))

    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    for epoch in range(num_epoch):
        print('this epoch')
        for i, batch in enumerate(train_loader):

            # Inputs T1-w and T2-w
            real_1 = Variable(batch["image_1"].type(Tensor))
            real_2 = Variable(batch["image_2"].type(Tensor))

            real_1[real_1 != real_1] = 0
            real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
            real_2[real_2 != real_2] = 0
            real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())

            ### Reshape input ###
            real_1 = F.interpolate(real_1, size=(input_dim, input_dim, input_dim), mode='trilinear', align_corners=False)
            real_2 = F.interpolate(real_2, size=(input_dim, input_dim, input_dim), mode='trilinear', align_corners=False)


            # Remove stored gradients
            optimizer.zero_grad()

            # Generate fake T2 images from the true T1 images
            fake_2 = generator(real_1)

            # Compute the corresponding loss
            loss = criterion(fake_2, real_2)

            # Compute the gradient and perform one optimization step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


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
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f] ETA: %s"
                % (
                    epoch + 1,
                    num_epoch,
                    i,
                    len(train_loader),
                    loss.item(),
                    time_left,
                )
            )


        columns = ['epoch', 'batch', 'loss']
        row = np.array(
            [epoch + 1, i,
             loss.item()]
        ).reshape(1, -1)
        row_df = pd.DataFrame(row, columns=columns)
        with open(filename, 'a') as f:
            row_df.to_csv(f, header=True, index=False, sep='\t')


        # Save results at the end of each epoch and images each 20 epochs
        if epoch % 20 == 0:
            sample_images(epoch)
        loss_valid = write_validation_tsv(epoch, test_loader, output_results, generator, criterion,
                                          input_dim)

        loss_is_best = loss_valid < best_valid_loss
        best_valid_loss = min(loss_valid, best_valid_loss)

        save_checkpoint({'model': generator.state_dict(),
                         'epoch': epoch,
                         'valid_loss': loss_valid},
                        loss_is_best,
                        model_dir)
        # Save optimizer state_dict to be able to reload
        save_checkpoint({'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'name': optimizer,
                         },
                        False,
                        model_dir,
                        filename='optimizer.pth.tar')
        del loss
        optimizer.zero_grad()
    return generator

def train_cyclegan(train_loader, test_loader, output_results,
                   caps_dir,
                   num_epoch=500,
                   lr=0.0001, beta1=0.9, beta2=0.999):
    """Train a CycleGAN.

    Args:
        train_loader: (DataLoader) a DataLoader wrapping a the training dataset
        test_loader: (DataLoader) a DataLoader wrapping a the test dataset
        num_epoch: (int) number of epochs performed during training
        lr: (float) learning rate of the discriminator and generator Adam optimizers
        beta1: (float) beta1 coefficient of the discriminator and generator Adam optimizers
        beta2: (float) beta1 coefficient of the discriminator and generator Adam optimizers

    Returns:
        generator: (nn.Module) the generator generating T2-w images from T1-w images.
    """
    columns = ['epoch', 'batch', 'loss_generator_from_1_to_2',
               'loss_generator_from_2_to_1',
               'loss_discriminator_from_1_to_2',
               'loss_discriminator_from_2_to_1']
    filename = os.path.join(output_results, 'training.tsv')

    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Output folder
    if not os.path.exists("./images/cyclegan"):
        os.makedirs("./images/cyclegan")

    # Loss functions
    criterion_GAN_from_1_to_2 = torch.nn.BCEWithLogitsLoss()  # A loss adapted to binary classification like torch.nn.BCEWithLogitsLoss
    criterion_GAN_from_2_to_1 = torch.nn.BCEWithLogitsLoss()  # A loss adapted to binary classification like torch.nn.BCEWithLogitsLoss
    criterion_pixelwise_from_1_to_2 = torch.nn.L1Loss()  # A loss for a voxel-wise comparison of images like torch.nn.L1Loss
    criterion_pixelwise_from_2_to_1 = torch.nn.L1Loss()  # A loss for a voxel-wise comparison of images like torch.nn.L1Loss

    lambda_GAN = 1.  # Weights criterion_GAN in the generator loss
    lambda_pixel = 1.  # Weights criterion_pixelwise in the generator loss

    # Initialize generators and discriminators
    generator_from_1_to_2 = GeneratorUNet()
    generator_from_2_to_1 = GeneratorUNet()
    discriminator_from_1_to_2 = DiscriminatorCycle()
    discriminator_from_2_to_1 = DiscriminatorCycle()

    if cuda:
        generator_from_t1_to_t2 = generator_from_1_to_2.cuda()
        generator_from_t2_to_t1 = generator_from_2_to_1.cuda()

        discriminator_from_t1_to_t2 = discriminator_from_1_to_2.cuda()
        discriminator_from_t2_to_t1 = discriminator_from_2_to_1.cuda()

        criterion_GAN_from_t1_to_t2 = criterion_GAN_from_1_to_2.cuda()
        criterion_GAN_from_t2_to_t1 = criterion_GAN_from_2_to_1.cuda()

        criterion_pixelwise_from_t1_to_t2 = criterion_pixelwise_from_1_to_2.cuda()
        criterion_pixelwise_from_t2_to_t1 = criterion_pixelwise_from_2_to_1.cuda()

    # Optimizers
    optimizer_generator_from_1_to_2 = torch.optim.Adam(
        generator_from_1_to_2.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_generator_from_2_to_1 = torch.optim.Adam(
        generator_from_2_to_1.parameters(), lr=lr, betas=(beta1, beta2))

    optimizer_discriminator_from_1_to_2 = torch.optim.Adam(
        discriminator_from_1_to_2.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_discriminator_from_2_to_1 = torch.optim.Adam(
        discriminator_from_2_to_1.parameters(), lr=lr, betas=(beta1, beta2))

    def sample_images(epoch):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(test_loader))
        real_1 = imgs["image_1"].type(Tensor)
        real_2 = imgs["image_2"].type(Tensor)

        ### reshape image
        real_1 = F.interpolate(real_1, size=(128, 128, 128), mode='trilinear', align_corners=False)
        real_2 = F.interpolate(real_2, size=(128, 128, 128), mode='trilinear', align_corners=False)

        real_1[real_1 != real_1] = 0
        real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
        real_2[real_2 != real_2] = 0
        real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())

        fake_2 = generator_from_1_to_2(real_1)
        img_nifti = os.path.join(caps_dir, 'subjects', imgs['participant_id'][0], imgs['session_id_2'][0],
                                 't1_linear',
                                 imgs['participant_id'][0] + '_' + imgs['session_id_2'][
                                     0] + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz')

        header = nib.load(img_nifti).header
        affine = nib.load(img_nifti).affine
        fake_2 = fake_2.detach().cpu().numpy()
        fake_2_example = nib.Nifti1Image(fake_2[0, 0, :, :, :], affine=affine, header=header)
        # if not os.path.exists(os.path.join(output_results, 'epoch-' + str(epoch))):
        #    os.makedirs(os.path.join(output_results, 'epoch-' + str(epoch)))
        fake_2_example.to_filename(os.path.join(output_results, 'epoch-' + str(epoch) + '_' +
                                                imgs['participant_id'][0] + '_' + imgs['session_id_2'][
                                                    0] + '_reconstructed.nii.gz'))

    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    for epoch in range(num_epoch):
        for i, batch in enumerate(train_loader):

            # Inputs T1-w and T2-w
            real_1 = batch["image_1"].type(Tensor)
            real_2 = batch["image_2"].type(Tensor)

            real_1[real_1 != real_1] = 0
            real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
            real_2[real_2 != real_2] = 0
            real_2 = (real_2 - real_2.min()) / (real_2.max() - real_2.min())

            ### Reshape input ###
            real_1 = F.interpolate(real_1, size=(128, 128, 128), mode='trilinear', align_corners=False)
            real_2 = F.interpolate(real_2, size=(128, 128, 128), mode='trilinear', align_corners=False)


            # Create labels
            valid_1 = Tensor(np.ones((real_1.size(0), 1, 1, 1, 1)))
            imitation_1 = Tensor(np.zeros((real_1.size(0), 1, 1, 1,1)))

            valid_2 = Tensor(np.ones((real_2.size(0), 1, 1, 1, 1)))
            imitation_2 = Tensor(np.zeros((real_2.size(0), 1, 1, 1, 1)))

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_generator_from_1_to_2.zero_grad()
            optimizer_generator_from_2_to_1.zero_grad()

            # GAN loss
            fake_2 = generator_from_1_to_2(real_1)
            pred_fake_2 = discriminator_from_1_to_2(fake_2)
            loss_GAN_from_1_to_2 = criterion_GAN_from_1_to_2(pred_fake_2, valid_2)

            fake_1 = generator_from_2_to_1(real_2)
            pred_fake_1 = discriminator_from_2_to_1(fake_1)
            loss_GAN_from_2_to_1 = criterion_GAN_from_2_to_1(pred_fake_1, valid_1)

            # L1 loss
            fake_fake_1 = generator_from_2_to_1(fake_2)
            loss_pixel_from_1_to_2 = criterion_pixelwise_from_t1_to_t2(fake_fake_1, real_1)

            fake_fake_2 = generator_from_t1_to_t2(fake_1)
            loss_pixel_from_2_to_1 = criterion_pixelwise_from_t2_to_t1(fake_fake_2, real_2)

            # Total loss
            loss_generator_from_1_to_2 = (lambda_GAN * loss_GAN_from_1_to_2 +
                                            lambda_pixel * loss_pixel_from_1_to_2)
            loss_generator_from_2_to_1 = (lambda_GAN * loss_GAN_from_2_to_1 +
                                            lambda_pixel * loss_pixel_from_2_to_1)

            loss_generator_from_1_to_2.backward()
            loss_generator_from_2_to_1.backward()

            optimizer_generator_from_1_to_2.step()
            optimizer_generator_from_2_to_1.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_discriminator_from_1_to_2.zero_grad()
            optimizer_discriminator_from_2_to_1.zero_grad()

            # Real loss
            pred_real_2 = discriminator_from_1_to_2(real_2)
            loss_real_2 = criterion_GAN_from_1_to_2(pred_real_2, valid_2)

            pred_real_1 = discriminator_from_2_to_1(real_1)
            loss_real_1 = criterion_GAN_from_2_to_1(pred_real_1, valid_1)

            # Fake loss
            pred_fake_2 = discriminator_from_1_to_2(fake_2.detach())
            loss_fake_2 = criterion_GAN_from_1_to_2(pred_fake_2, imitation_2)

            pred_fake_1 = discriminator_from_2_to_1(fake_1.detach())
            loss_fake_1 = criterion_GAN_from_2_to_1(pred_fake_1, imitation_1)

            # Total loss
            loss_discriminator_from_1_to_2 = 0.5 * (loss_real_2 + loss_fake_2)
            loss_discriminator_from_2_to_1 = 0.5 * (loss_real_1 + loss_fake_1)

            loss_discriminator_from_1_to_2.backward()
            loss_discriminator_from_2_to_1.backward()

            optimizer_discriminator_from_1_to_2.step()
            optimizer_discriminator_from_2_to_1.step()

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
                "\r[Epoch %d/%d] [Batch %d/%d] "
                "[Generator losses: %f, %f] "
                "[Discriminator losses: %f, %f] "
                "ETA: %s"
                % (
                    epoch + 1,
                    num_epoch,
                    i,
                    len(train_loader),
                    loss_generator_from_1_to_2.item(),
                    loss_generator_from_2_to_1.item(),
                    loss_discriminator_from_1_to_2.item(),
                    loss_discriminator_from_2_to_1.item(),
                    time_left,
                )
            )

        columns = ['epoch', 'batch', 'loss_generator_from_1_to_2',
                   'loss_generator_from_2_to_1',
                   'loss_discriminator_from_1_to_2',
                   'loss_discriminator_from_2_to_1']
        row = np.array(
            [epoch + 1, i, loss_generator_from_1_to_2.item(), loss_generator_from_2_to_1.item(),
            loss_discriminator_from_1_to_2.item(),
             loss_discriminator_from_2_to_1.item()]
        ).reshape(1, -1)
        row_df = pd.DataFrame(row, columns=columns)
        with open(filename, 'a') as f:
            row_df.to_csv(f, header=True, index=False, sep='\t')


        # Save images at the end of each epoch
        sample_images(epoch)

    return generator_from_t1_to_t2