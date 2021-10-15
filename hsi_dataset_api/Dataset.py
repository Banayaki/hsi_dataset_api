import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import yaml

from .utils import WrongDirectoryStructure


@dataclass
class HsiDatapoint:
    hsi: Tuple[np.ndarray, str]
    mask: Tuple[np.ndarray, str]
    meta: Tuple[dict, str]


class HsiDataset:
    def __init__(self, path_to_dataset: str) -> None:
        """

        :param path_to_dataset:
        """
        self._check_folder_structure(path_to_dataset)
        self.path_to_dataset = path_to_dataset

        self.dataset_description = self._read_dataset_description()

        self.train_hsi_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'train', 'hsi', '*.npy')))
        self.train_meta_hsi_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'train', 'hsi', '*.yml')))
        self.train_masks_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'train', 'masks', '*.png')))
        self.train_set_length = len(self.train_hsi_filenames)

        self.test_hsi_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'test', 'hsi', '*.npy')))
        self.test_meta_hsi_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'test', 'hsi', '*.yml')))
        self.test_masks_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'test', 'masks', '*.png')))
        self.test_set_length = len(self.test_hsi_filenames)

    def _check_folder_structure(self, path_to_dataset: str):
        """

        :param path_to_dataset:
        :return:
        """
        msg = 'Wrong dataset directory structure.'

        required_files = {'test', 'train', 'meta.yml'}
        if len(set(os.listdir(path_to_dataset)) - required_files) != 0:
            raise WrongDirectoryStructure(msg + f"In {path_to_dataset}")

        for data_folder in ['test', 'train']:
            data_path = os.path.join(path_to_dataset, data_folder)
            if len(set(os.listdir(data_path)) - {'hsi', 'masks'}) != 0:
                raise WrongDirectoryStructure(msg + f'In {data_path}')

            hsi_data_path = os.path.join(data_path, 'hsi')
            masks_data_path = os.path.join(data_path, 'masks')

            if len(os.listdir(hsi_data_path)) // 2 != len(os.listdir(masks_data_path)):
                raise WrongDirectoryStructure(f'Number of HSIs and masks are not equal in folder {data_path}')

    def _read_yml(self, path):
        with open(path, 'r') as f:
            return yaml.load(f)

    def _read_dataset_description(self):
        return self._read_yml(os.path.join(self.path_to_dataset, 'meta.yml'))

    def get_dataset_description(self) -> dict:
        return self.dataset_description

    def data_iterator(self, opened: bool = True, train: bool = False, shuffle: bool = False):
        """
        RAM friendly method that returns a generator of HSI data points

        :param shuffle: if TRUE returns data in shuffled way
        :param opened: if TRUE returns opened data points i.e np.ndarrays instead paths to the data
        :param train: if TRUE train data returns
        :return: generator with return HsiDatapoint
        """
        if train:
            number_of_samples = self.train_set_length
            hsi_filenames = self.train_hsi_filenames
            meta_hsi_filenames = self.train_meta_hsi_filenames
            mask_filenames = self.train_masks_filenames
        else:
            number_of_samples = self.test_set_length
            hsi_filenames = self.test_hsi_filenames
            meta_hsi_filenames = self.test_meta_hsi_filenames
            mask_filenames = self.test_masks_filenames

        choose_array = np.random.permutation(number_of_samples) if shuffle else np.arange(number_of_samples)
        for elem_idx in choose_array:
            hsi_filename = hsi_filenames[elem_idx]
            meta_hsi_filename = meta_hsi_filenames[elem_idx]
            mask_filename = mask_filenames[elem_idx]

            if opened:
                hsi_datapoint = HsiDatapoint(
                    hsi=np.load(hsi_filename),
                    mask=cv2.imread(mask_filename),
                    meta=self._read_yml(meta_hsi_filename)
                )
            else:
                hsi_datapoint = HsiDatapoint(
                    hsi=hsi_filename,
                    mask=mask_filename,
                    meta=meta_hsi_filename
                )

            yield hsi_datapoint

    def data_readall_as_list(self, opened: bool = True, train: bool = False, shuffle: bool = False) \
            -> List[HsiDatapoint]:
        """
        Returns list of objects, RAM consuming if opened flag set TRUE

        :param shuffle: if TRUE returns data in shuffled way
        :param opened: if TRUE returns opened data points i.e np.ndarrays instead paths to the data
        :param train: if TRUE train data returns
        :return: list with HsiDatapoints
        """
        data = []
        for hsi_datapoint in self.data_iterator(opened, train, shuffle):
            data.append(hsi_datapoint)
        return data
