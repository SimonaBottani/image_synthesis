import pandas as pd
import os.path as path
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import abc

FILENAME_TYPE = {'full': '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w',
                 'cropped': '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w'}


def commandline_to_json(commandline, test=False):
    """
    This is a function to write the python argparse object into a json file.
    This helps for DL when searching for hyperparameters

    :param commandline: a tuple contain the output of
                        `parser.parse_known_args()`

    :return:
    """
    import json
    import os

    commandline_arg_dic = vars(commandline[0])
    commandline_arg_dic['unknown_arg'] = commandline[1]

    output_dir = commandline_arg_dic['output_results']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save to json file
    json = json.dumps(commandline_arg_dic, skipkeys=True, indent=4)
    if test:
        print("Path of json file:", os.path.join(output_dir, "commandline_test.json"))
        f = open(os.path.join(output_dir, "commandline_test.json"), "w")
    else:
        print("Path of json file:", os.path.join(output_dir, "commandline.json"))
        f = open(os.path.join(output_dir, "commandline.json"), "w")
    f.write(json)
    f.close()


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


def find_image_path(caps_dir, participant_id, session_id, preprocessing):
    from os import path
    if preprocessing == "t1-linear":
        image_path = path.join(caps_dir, 'subjects', participant_id, session_id,
                               't1_linear',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['cropped'] + '.nii.gz')
    elif preprocessing == "t1-extensive":
        image_path = path.join(caps_dir, 'subjects', participant_id, session_id,
                               't1', 'spm', 'segmentation', 'normalized_space',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['skull_stripped'] + '.nii.gz')
    else:
        raise ValueError(
            "Preprocessing %s must be in ['t1-linear', 't1-extensive']." %
            preprocessing)

    return image_path


class ToTensor(object):
    """Convert image type to Tensor and diagnosis to diagnosis code"""

    def __call__(self, image):
        np.nan_to_num(image, copy=False)
        image = image.astype(float)

        return torch.from_numpy(image[np.newaxis, :]).float()


def get_transforms(mode, minmaxnormalization=True):
    if mode in ["image", "patch", "roi"]:
        if minmaxnormalization:
            transformations = MinMaxNormalization()
        else:
            transformations = None
    elif mode == "slice":
        trg_size = (224, 224)
        if minmaxnormalization:
            transformations = transforms.Compose([MinMaxNormalization(),
                                                  transforms.ToPILImage(),
                                                  transforms.Resize(trg_size),
                                                  transforms.ToTensor()])
        else:
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(trg_size),
                                                  transforms.ToTensor()])
    else:
        raise ValueError("Transforms for mode %s are not implemented." % mode)

    return transformations


def find_image_path(caps_dir, participant_id, session_id, preprocessing):
    from os import path
    if preprocessing == "t1-linear":
        image_path = path.join(caps_dir, 'subjects', participant_id, session_id,
                               't1_linear',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['cropped'] + '.nii.gz')
    elif preprocessing == "t1-extensive":
        image_path = path.join(caps_dir, 'subjects', participant_id, session_id,
                               't1', 'spm', 'segmentation', 'normalized_space',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['skull_stripped'] + '.nii.gz')
    else:
        raise ValueError(
            "Preprocessing %s must be in ['t1-linear', 't1-extensive']." %
            preprocessing)

    return image_path


class MRIDataset(Dataset):
    """Abstract class for all derived MRIDatasets."""

    def __init__(self, caps_directory, data_file,
                 preprocessing, transformations=None, skull_strip=None):
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.diagnosis_code = {
            'tier_1': 0,
            'tier_2': 1,
            'tier_3': 2,
            'tier_4': 3,
            'gaudo_0': 0,
            'gaudo_1': 1,
            'mov_0': 0,
            'mov_1': 1,
            'mov_2': 2,
            'noise_0': 0,
            'noise_1': 1,
            'noise_2': 2,
            'cont_0': 0,
            'cont_1': 1,
            'cont_2': 2,
            'unlabeled': -1}
        self.preprocessing = preprocessing
        self.skull_strip = skull_strip

        if not hasattr(self, 'elem_index'):
            raise ValueError(
                "Child class of MRIDataset must set elem_index attribute.")
        if not hasattr(self, 'mode'):
            raise ValueError(
                "Child class of MRIDataset must set mode attribute.")

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument data_file is not of correct type.')

        mandatory_col = {"participant_id", "session_id", "diagnosis_1"}
        #if self.elem_index == "mixed":
        #    mandatory_col.add("%s_id" % self.mode)

        #if not mandatory_col.issubset(set(self.df.columns.values)):
        #    raise Exception("the data file is not in the correct format."
        #                    "Columns should include %s" % mandatory_col)

        self.elem_per_image = self.num_elem_per_image()

    def __len__(self):
        return len(self.df) * self.elem_per_image

    def _get_path(self, participant, session, mode="image"):

        if self.preprocessing == "t1-linear":
            if self.skull_strip != 'skull_strip':
                image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_linear',
                                   participant + '_' + session
                                   + FILENAME_TYPE['cropped'] + '.pt')
            elif self.skull_strip == "skull_strip":
                image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                       'deeplearning_prepare_data', '%s_based' % mode, 't1_linear',
                                       participant + '_' + session
                                       + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_skull-skripped-hdbet_T1w.pt')
        elif self.preprocessing == "t1-extensive":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_extensive',
                                   participant + '_' + session
                                   + FILENAME_TYPE['skull_stripped'] + '.pt')
        else:
            raise NotImplementedError(
                "The path to preprocessing %s is not implemented" % self.preprocessing)

        return image_path

    def _get_meta_data(self, idx):
        image_idx = idx // self.elem_per_image
        participant = self.df.loc[image_idx, 'participant_id']
        session = self.df.loc[image_idx, 'session_id']


        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        elif self.elem_index == "mixed":
            elem_idx = self.df.loc[image_idx, '%s_id' % self.mode]
        else:
            elem_idx = self.elem_index

        diagnosis = self.df.loc[image_idx, 'diagnosis']
        label = self.diagnosis_code[diagnosis]

        return participant, session, elem_idx, label

    def _get_meta_data_paired_images(self, idx):
        image_idx = idx // self.elem_per_image
        participant = self.df.loc[image_idx, 'participant_id']
        session_1 = self.df.loc[image_idx, 'session_id_1']
        session_2 = self.df.loc[image_idx, 'session_id_2']
        label_1 = self.df.loc[image_idx, 'diagnosis_1']
        label_2 = self.df.loc[image_idx, 'diagnosis_2']

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        elif self.elem_index == "mixed":
            elem_idx = self.df.loc[image_idx, '%s_id' % self.mode]
        else:
            elem_idx = self.elem_index


        #label = self.diagnosis_code[diagnosis]

        return participant, session_1, session_2, elem_idx, label_1, label_2

    def _get_full_image(self):
        import nibabel as nib

        participant_id = self.df.loc[0, 'participant_id']
        session_id = self.df.loc[0, 'session_id']

        try:
            image_path = self._get_path(participant_id, session_id, "image")
            image = torch.load(image_path)
            image[image != image] = 0
            image = (image - image.min()) / (image.max() - image.min())
            print(torch.max(torch.reshape(image, (-1,))))
            print(torch.isnan(image).any())

        except FileNotFoundError:
            image_path = find_image_path(
                self.caps_directory,
                participant_id,
                session_id,
                preprocessing=self.preprocessing)
            image_nii = nib.load(image_path)
            image_np = image_nii.get_fdata()
            image = ToTensor()(image_np)

        return image

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    @abc.abstractmethod
    def num_elem_per_image(self):
        pass


class MRIDatasetImage(MRIDataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self, caps_directory, data_file,
                 preprocessing='t1-linear', transformations=None, skull_strip=None):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            transformations (callable, optional): Optional transform to be applied on a sample.

        """
        self.elem_index = None
        self.mode = "image"
        super().__init__(caps_directory, data_file, preprocessing, transformations, skull_strip)

    def __getitem__(self, idx):
        participant, session_1, session_2, _, label_1, label_2 = self._get_meta_data_paired_images(idx)

        image_path_1 = self._get_path(participant, session_1, "image")
        image_path_1_skull_strip = path.join(self.caps_directory, 'subjects', participant, session_1,
                               'deeplearning_prepare_data', '%s_based' % 'image', 't1_linear',
                               participant + '_' + session_1
                               + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_skull-skripped-hdbet_T1w.pt')
        print(image_path_1)

        image_1 = torch.load(image_path_1)
        image_path_1_skull_strip = torch.load(image_path_1_skull_strip)

        image_path_2 = self._get_path(participant, session_2, "image")
        image_path_2_skull_strip = path.join(self.caps_directory, 'subjects', participant, session_2,
                               'deeplearning_prepare_data', '%s_based' % 'image', 't1_linear',
                               participant + '_' + session_2
                               + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_skull-skripped-hdbet_T1w.pt')

        print(image_path_2)
        image_2 = torch.load(image_path_2)
        image_path_2_skull_strip = torch.load(image_path_2_skull_strip)

        if self.transformations:
            image_1 = self.transformations(image_1)
            image_2 = self.transformations(image_2)

        sample = {'image_1': image_1, 'label_1': label_1,
                  'participant_id': participant,
                  'session_id_1': session_1,
                  'image_path_1': image_path_1,
                  'image_2': image_2, 'label_2': label_2,
                  'session_id_2': session_2,
                  'image_path_2': image_path_2,
                  'image_path_1_skull_strip':image_path_1_skull_strip,
                  'image_path_2_skull_strip': image_path_2_skull_strip}


        return sample

    def num_elem_per_image(self):
        return 1


def load_data(train_val_path, diagnoses_list,
              split, n_splits=None, baseline=True):

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()

    if n_splits is None:
        train_path = path.join(train_val_path, 'train')
        valid_path = path.join(train_val_path, 'validation')

    else:
        train_path = path.join(train_val_path, 'train_splits-' + str(n_splits),
                               'split-' + str(split))
        valid_path = path.join(train_val_path, 'validation_splits-' + str(n_splits),
                               'split-' + str(split))

    print("Train", train_path)
    print("Valid", valid_path)

    for diagnosis in diagnoses_list:

        if baseline:
            train_diagnosis_path = path.join(
                train_path, diagnosis + '_baseline.tsv')
        else:
            train_diagnosis_path = path.join(train_path, diagnosis + '.tsv')

        valid_diagnosis_path = path.join(
            valid_path, diagnosis + '_baseline.tsv')

        train_diagnosis_df = pd.read_csv(train_diagnosis_path, sep='\t')
        valid_diagnosis_df = pd.read_csv(valid_diagnosis_path, sep='\t')

        train_df = pd.concat([train_df, train_diagnosis_df])
        valid_df = pd.concat([valid_df, valid_diagnosis_df])

    train_df.reset_index(inplace=True, drop=True)
    valid_df.reset_index(inplace=True, drop=True)

    return train_df, valid_df




def save_checkpoint(state, loss_is_best, checkpoint_dir, filename='checkpoint.pth.tar',
                   best_loss='best_loss'):
    import torch
    import os
    import shutil

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save(state, os.path.join(checkpoint_dir, filename))

    if loss_is_best:
        best_loss_path = os.path.join(checkpoint_dir, best_loss)
        if not os.path.exists(best_loss_path):
            os.makedirs(best_loss_path)
        shutil.copyfile(os.path.join(checkpoint_dir, filename), os.path.join(best_loss_path, 'model_best.pth.tar'))


def load_model(model, checkpoint_dir, gpu, filename='model_best.pth.tar'):
    """
    Load the weights written in checkpoint_dir in the model object.

    :param model: (Module) CNN in which the weights will be loaded.
    :param checkpoint_dir: (str) path to the folder containing the parameters to loaded.
    :param gpu: (bool) if True a gpu is used.
    :param filename: (str) Name of the file containing the parameters to loaded.
    :return: (Module) the update model.
    """
    from copy import deepcopy
    import torch
    import os

    best_model = deepcopy(model)
    param_dict = torch.load(os.path.join(checkpoint_dir, filename), map_location="cpu")
    best_model.load_state_dict(param_dict['model'])

    if gpu:
        best_model = best_model.cuda()

    return best_model, param_dict['epoch']


def load_optimizer(optimizer_path, model):
    """
    Creates and load the state of an optimizer.

    :param optimizer_path: (str) path to the optimizer.
    :param model: (Module) model whom parameters will be optimized by the created optimizer.
    :return: optimizer initialized with specific state and linked to model parameters.
    """
    from os import path
    import torch

    if not path.exists(optimizer_path):
        raise ValueError('The optimizer was not found at path %s' % optimizer_path)
    print('Loading optimizer')
    optimizer_dict = torch.load(optimizer_path)
    name = optimizer_dict["name"]
    optimizer = eval("torch.optim." + name)(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer.load_state_dict(optimizer_dict["optimizer"])


def extract_patch_tensor(
    image_tensor: torch.Tensor,
    patch_size: int,
    stride_size: int,
    patch_index: int,
    patches_tensor: torch.Tensor = None,
) -> torch.Tensor:
    """Extracts a single patch from image_tensor"""

    if patches_tensor is None:
        patches_tensor = (
            image_tensor.unfold(1, patch_size, stride_size)
            .unfold(2, patch_size, stride_size)
            .unfold(3, patch_size, stride_size)
            .contiguous()
        )

        # the dimension of patches_tensor is [1, patch_num1, patch_num2, patch_num3, patch_size1, patch_size2, patch_size3]
        patches_tensor = patches_tensor.view(-1, patch_size, patch_size, patch_size)
        print(patches_tensor.shape)

    return patches_tensor[patch_index, ...].unsqueeze_(0).clone()
