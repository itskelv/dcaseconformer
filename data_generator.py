"""
data_generator.py

This module handles the creation of data generators for efficient data loading and preprocessing during training.

Class:
    DataGenerator: A data generator for efficient data loading and preprocessing during training.

Methods:
    __init__(self, params, mode='dev_train'): Initializes the DataGenerator instance.
    __getitem__(self, item): Returns the data for a given index.
    __len__(self): Returns the number of data points.
    get_feature_files(self): Collects the paths to the feature files based on the selected folds and modality.
    get_folds(self): Returns the folds for the given data split.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: February 2025
"""

import os
import torch
import glob
from torch.utils.data.dataset import Dataset
import numpy as np


class DataGenerator(Dataset):
    def __init__(self, params, mode='dev_train'):
        """
        Initializes the DataGenerator instance.
        Args:
            params (dict): Parameters for data generation.
            mode (str): data split ('dev_train', 'dev_test').
        """

        super().__init__()
        self.params = params
        self.mode = mode
        self.root_dir = params['root_dir']
        self.feat_dir = params['feat_dir']
        self.modality = params['modality']

        self.folds = self.get_folds()

        # self.video_files will be an empty [] if self.modality == 'audio'
        self.audio_files, self.video_files, self.label_files = self.get_feature_files()

    def __getitem__(self, item):
        """
        Returns the data for a given index.
        Args:
            item (int): Index of the data.
        Returns:
            tuple: A tuple containing audio features, video_features (for audio_visual modality), and labels.
        """
        # Load the features that are always present.
        audio_file = self.audio_files[item]
        audio_features = torch.load(audio_file)

        # --- CHECK THE MODE FIRST ---
        if self.mode == 'eval':
            # In eval mode, there are no labels. Return a dummy label.
            dummy_labels = torch.zeros(1)  # Or whatever shape is compatible downstream
            if self.modality == 'audio_visual':
                video_file = self.video_files[item]
                video_features = torch.load(video_file)
                return (audio_features, video_features), dummy_labels
            return audio_features, dummy_labels

        # --- If not in 'eval' mode, this code will run for 'dev_train' and 'dev_test' ---
        label_file = self.label_files[item]
        labels = torch.load(label_file)

        if not self.params['multiACCDOA']:
            mask = labels[:, :self.params['nb_classes']]
            mask = mask.repeat(1, 4)
            labels = mask * labels[:, self.params['nb_classes']:]

        if self.modality == 'audio_visual':
            video_file = self.video_files[item]
            video_features = torch.load(video_file)
            return (audio_features, video_features), labels
        else:
            if self.params['multiACCDOA']:
                labels = labels[:, :, :-1, :]
            else:
                labels = labels[:, :-self.params['nb_classes']]
            return audio_features, labels

    def __len__(self):
        """
        Returns the number of data points.
        Returns:
            int: Number of data points.
        """
        #return 100
        return len(self.audio_files)

    def get_feature_files(self):
        """
        Collects the paths to the feature and label files based on the selected folds and modality.
        Returns:
            tuple: A tuple containing lists of paths to audio feature files, video feature files, and processed label files.
        """
        audio_files, video_files, label_files = [], [], []
        print(f"Feature dir being used: {self.feat_dir}")
        print(f"Mode: {self.mode}")
        print(f"Folds: {self.folds}")


        # Loop through each fold and collect files
        if self.mode == 'eval':
            audio_files = glob.glob(os.path.join(self.feat_dir, 'stereo_eval/*.pt'))
            if self.modality == 'audio_visual':
                video_files = glob.glob(os.path.join(self.feat_dir, 'video_eval/*.pt'))
            # label_files remains empty for eval mode

        else:  # Original logic for 'dev_train' and 'dev_test'
            for fold in self.folds:
                audio_files += glob.glob(os.path.join(self.feat_dir, f'stereo_dev/*.pt'))
                folder = 'metadata_dev_adpit' if self.params['multiACCDOA'] else 'metadata_dev'
                label_files += glob.glob(os.path.join(self.feat_dir, folder, f'*{fold}*.pt'))
                if self.modality == 'audio_visual':
                    video_files += glob.glob(os.path.join(self.feat_dir, f'video_dev/*.pt'))

        # Sort files to ensure corresponding audio, video, and label files are in the same order
        audio_files = sorted(audio_files, key=lambda x: x.split('/')[-1])
        if self.modality == 'audio_visual':
            video_files = sorted(video_files, key=lambda x: x.split('/')[-1])
        if self.mode != 'eval':
            label_files = sorted(label_files, key=lambda x: x.split('/')[-1])

        return audio_files, video_files, label_files

    def get_folds(self):
        """
        Returns the folds for the given data split
        Returns:
            list: List of folds.
        """
        if self.mode == 'dev_train':
            return self.params['dev_train_folds']
        elif self.mode == 'dev_test':
            return self.params['dev_test_folds']
        elif self.mode == 'eval': # No folds for evaluation set
            return []
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from ['dev_train', 'dev_test', 'eval'].")



if __name__ == '__main__':
    # use this space to test if the DataGenerator class works as expected.
    # All the classes will be called from the main.py for actual use.

    from parameters import params
    from torch.utils.data import DataLoader
    params['multiACCDOA'] = False
    dev_train_dataset = DataGenerator(params=params, mode='dev_train')
    dev_train_iterator = DataLoader(dataset=dev_train_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=params['shuffle'], pin_memory=False,  drop_last=True)

    for i, (input_features, labels) in enumerate(dev_train_iterator):
        if params['modality'] == 'audio':
            print(input_features.size())
            print(labels.size())
        elif params['modality'] == 'audio_visual':
            print(input_features[0].size())
            print(input_features[1].size())
            print(labels.size())
        break



