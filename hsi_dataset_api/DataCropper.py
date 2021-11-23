import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
import cv2
from numba import jit

import os
import pickle
from typing import List, Optional


@jit(nopython=True)
def numba_check_ratio_function(mask, ih, iw, STEP, PART_SIZE, ratio):
    part = mask[ih * STEP: ih * STEP + PART_SIZE, iw * STEP: iw * STEP + PART_SIZE]
    total_pixels = np.prod(part.shape)
    positive_pixels = np.sum(part)
    return (positive_pixels / total_pixels) > ratio


@jit(parallel=True)
def numba_find_crop_areas_loop(mask, class_masks, objects_ratio, min_class_ratio, matrix, class_matrices, nh, nw, STEP,
                               PART_SIZE):
    for ih in range(nh):
        for iw in range(nw):
            matrix[ih, iw] = numba_check_ratio_function(mask, ih, iw, STEP, PART_SIZE, objects_ratio)

            for i in range(len(class_masks)):
                class_matrices[i, ih, iw] = numba_check_ratio_function(class_masks[i], ih, iw, STEP, PART_SIZE,
                                                                       min_class_ratio)


class HsiDataCropper:
    def __init__(self,
                 side_size: int = 512,
                 step: int = 8,
                 objects_ratio: float = 0.75,
                 min_class_ratio: float = 0.25,
                 only_single_class: bool = False):
        """
        :param side_size: a size of a square of the specter to be created
        :param step: defines a step for sliding window movement, it must be a divider for the side_size
        :param objects_ratio: percent of pixels that do not belong to the background class
        :param min_class_ratio: minimal percentage of pixels that belongs to a single class (not a bkg)
        :param only_single_class: flags that specter must contain one and only one class
        """
        self.side_size = side_size
        self.step = step
        self.objects_ratio = objects_ratio
        self.min_class_ratio = min_class_ratio
        self.only_single_class = only_single_class

        self.classes2quantity = None
        self.classes2area = None

    def _reset(self, classes: List[str]):
        self.classes2quantity = {_cls: 0 for _cls in classes}
        self.classes2area = {_cls: 0 for _cls in classes}

    def draw_statistics(self, output_filename: str):
        pass

    def crop(self, base_path: str, output_path: str, classes: List[str], selected_folders: Optional[List[str]] = None,
             save_statistics: bool = False, threshold: int = 128):
        STEP = self.step
        PART_SIZE = self.side_size
        specters_output_path = os.path.join(output_path, 'hsi')
        masks_output_path = os.path.join(output_path, 'masks')
        os.makedirs(specters_output_path, exist_ok=True)
        os.makedirs(masks_output_path, exist_ok=True)

        if selected_folders is None:
            selected_folders = os.listdir(base_path)

        for folder_name in selected_folders:
            path_to_data = os.path.join(base_path, folder_name)
            fnames = os.listdir(path_to_data)
            mask = None
            class_masks = []
            has_classes = []
            specter = np.load(glob.glob(os.path.join(path_to_data, '*.npy'))[0])

            for fname in fnames:
                for _cls in classes:
                    if _cls in fname:
                        image = cv2.imread(os.path.join(path_to_data, fname))
                        class_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > threshold
                        class_masks.append(class_mask)
                        has_classes.append(_cls)
                        if mask is None:
                            mask = class_mask
                        else:
                            mask += class_mask

            if self.only_single_class and len(has_classes) > 0:
                print(folder_name, 'Contains more than one class. SKIPPING...')
                continue

            nh = (mask.shape[0] - PART_SIZE) // STEP
            nw = (mask.shape[1] - PART_SIZE) // STEP
            print('Cropping with following parameters')
            print(f'PART_SIZE = {PART_SIZE}')
            print(f'STEP = {STEP}')
            print('Initial specter shape: ', mask.shape)
            print('nh = ', nh, ' nw = ', nw)

            matrix = np.zeros((nh, nw))
            class_matrices = np.zeros((len(has_classes), nh, nw))

            numba_find_crop_areas_loop(matrix, class_matrices)

            for iy in range(matrix.shape[0]):
                for ix in range(matrix.shape[1]):
                    matrix_part = matrix[iy: iy + PART_SIZE // STEP, ix: ix + PART_SIZE // STEP]
                    if np.all(matrix_part):
                        # print(folder_name, ': Taking', (iy, ix), 'position for specter')
                        matrix[iy: iy + PART_SIZE // STEP, ix: ix + PART_SIZE // STEP] = False
                        part = specter[iy * STEP: iy * STEP + PART_SIZE * STEP]
                        mask = []

                        for _cls, cl_mat in zip(has_classes, class_masks):
                            class_part = cl_mat[iy * STEP: iy * STEP + PART_SIZE * STEP]
                            if np.any(class_part):
                                if save_statistics:
                                    self.classes2quantity[_cls] += 1
                                    self.classes2area[_cls] += np.sum(class_part)

                            class_value = classes.index(_cls) + 1
                            class_part[class_part is True] = class_value
                            mask.append(class_part)
                        mask = np.array(mask)

                        np.save(os.path.join(specters_output_path, f'{folder_name}.npy'), part)
                        cv2.imwrite(os.path.join(masks_output_path, f'{folder_name}.png'), mask)
