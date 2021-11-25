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
    def __init__(self, path_to_dataset: str, cropped_dataset: bool = False) -> None:
        """
        :param path_to_dataset: path to the root folder of the dataset
        :param cropped_dataset: cropped dataset stores into a different structure, where an additional level of folders
        are exist.
        """
        self._check_folder_structure(path_to_dataset)
        self.path_to_dataset = path_to_dataset

        self.dataset_description = self._read_dataset_description()

        if cropped_dataset:
            self.hsi_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'hsi', '*', '*.npy')))
            self.meta_hsi_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'hsi', '*', '*.yml')))
            self.masks_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'masks', '*', '*.png')))
        else:
            self.hsi_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'hsi', '*.npy')))
            self.meta_hsi_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'hsi', '*.yml')))
            self.masks_filenames = sorted(glob.glob(os.path.join(self.path_to_dataset, 'masks', '*.png')))

        self.set_length = len(self.hsi_filenames)

    def _check_folder_structure(self, path_to_dataset: str):
        """

        :param path_to_dataset:
        :return:
        """
        msg = 'Wrong dataset directory structure.'

        required_files = {'hsi', 'masks', 'meta.yml'}
        if len(set(os.listdir(path_to_dataset)) - required_files) != 0:
            raise WrongDirectoryStructure(msg + f"In {path_to_dataset}")

    def _read_yml(self, path):
        with open(path, 'r') as f:
            return yaml.full_load(f)

    def _read_dataset_description(self):
        return self._read_yml(os.path.join(self.path_to_dataset, 'meta.yml'))

    def get_dataset_description(self) -> dict:
        return self.dataset_description

    def data_iterator(self, opened: bool = True, shuffle: bool = False):
        """
        RAM friendly method that returns a generator of HSI data points

        :param shuffle: if TRUE returns data in shuffled way
        :param opened: if TRUE returns opened data points i.e np.ndarrays instead paths to the data
        :return: generator with return HsiDatapoint
        """
        choose_array = np.random.permutation(self.set_length) if shuffle else np.arange(self.set_length)
        for elem_idx in choose_array:
            hsi_filename = self.hsi_filenames[elem_idx]
            meta_hsi_filename = self.meta_hsi_filenames[elem_idx]
            mask_filename = self.masks_filenames[elem_idx]

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

    def data_readall_as_list(self, opened: bool = True, shuffle: bool = False) \
            -> List[HsiDatapoint]:
        """
        Returns list of objects, RAM consuming if opened flag set TRUE

        :param shuffle: if TRUE returns data in shuffled way
        :param opened: if TRUE returns opened data points i.e np.ndarrays instead paths to the data
        :return: list with HsiDatapoints
        """
        data = []
        for hsi_datapoint in self.data_iterator(opened, shuffle):
            data.append(hsi_datapoint)
        return data
