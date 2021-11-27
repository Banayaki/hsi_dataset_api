import glob
import json
import os
from typing import List, Optional

import cv2
import numpy as np
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm
from numba import jit, prange


@jit(nopython=True)
def numba_check_ratio_function(mask, ih, iw, STEP, PART_SIZE, ratio):
    part = mask[ih * STEP: ih * STEP + PART_SIZE, iw * STEP: iw * STEP + PART_SIZE]
    total_pixels = part.shape[0] * part.shape[1]
    positive_pixels = np.sum(part)
    return (positive_pixels / total_pixels) > ratio


@jit(parallel=True)
def numba_find_crop_areas_loop(mask, class_masks, objects_ratio, min_class_ratio, matrix, class_matrices, nh, nw, STEP,
                               PART_SIZE):
    for ih in prange(nh):
        for iw in prange(nw):
            matrix[ih, iw] = numba_check_ratio_function(mask, ih, iw, STEP, PART_SIZE, objects_ratio)

            for i in prange(len(class_masks)):
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

    def draw_statistics(self, output_filename: Optional[str] = None, print_values: bool = False):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
        ax1.bar(range(len(self.classes2quantity.values())), list(self.classes2quantity.values()))
        ax1.set_xticks(range(len(self.classes2quantity.values())))
        ax1.set_xticklabels(list(self.classes2quantity.keys()), rotation=90)

        ax2.bar(range(len(self.classes2quantity.values())), list(self.classes2area.values()))
        ax2.set_xticks(range(len(self.classes2quantity.values())))
        ax2.set_xticklabels(list(self.classes2area.keys()), rotation=90)

        if print_values:
            for i, v in enumerate(self.classes2quantity.values()):
                ax1.text(i - 0.1, v + 0.2, str(v), color='red', fontweight='bold')

        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of images')
        ax1.set_title('Number of images containing class')

        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Number of pixels')
        ax2.set_title('Number of pixels containing class')

        if output_filename is not None:
            fig.savefig(f'{output_filename}.svg')

    def crop(self, base_path: str, output_path: str, classes: List[str], selected_folders: Optional[List[str]] = None,
             save_statistics: bool = False, threshold: int = 128):
        self._reset(classes)
        STEP = self.step
        PART_SIZE = self.side_size
        specters_output_path = os.path.join(output_path, 'hsi')
        masks_output_path = os.path.join(output_path, 'masks')
        os.makedirs(specters_output_path, exist_ok=True)
        os.makedirs(masks_output_path, exist_ok=True)

        if selected_folders is None:
            selected_folders = os.listdir(base_path)

        for folder_name in tqdm(selected_folders):
            path_to_data = os.path.join(base_path, folder_name)
            fnames = os.listdir(path_to_data)
            mask = None
            class_masks = []
            has_classes = []
            specter = np.load(glob.glob(os.path.join(path_to_data, '*.npy'))[0])
            # with open(os.path.join(path_to_data, 'meta.json'), 'r') as f:
            #     meta_info = json.load(f)

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

            class_masks = np.array(class_masks)
            numba_find_crop_areas_loop(mask, class_masks, self.objects_ratio, self.min_class_ratio, matrix,
                                       class_matrices, nh, nw, STEP,
                                       PART_SIZE)

            crop_counter = 0

            for iy in range(matrix.shape[0]):
                for ix in range(matrix.shape[1]):
                    part_has_class = []
                    matrix_part = matrix[iy: iy + PART_SIZE // STEP, ix: ix + PART_SIZE // STEP]
                    if np.all(matrix_part):
                        # print(folder_name, ': Taking', (iy, ix), 'position for specter')
                        matrix[iy: iy + PART_SIZE // STEP, ix: ix + PART_SIZE // STEP] = False
                        part = specter[:, iy * STEP: iy * STEP + PART_SIZE, ix * STEP: ix * STEP + PART_SIZE]
                        mask = np.zeros(part.shape[1:], dtype=np.uint8)

                        for idx, _cls in enumerate(has_classes):
                            class_matrix = class_matrices[idx, iy: iy + PART_SIZE // STEP, ix: ix + PART_SIZE // STEP]
                            if np.all(class_matrix):
                                class_part = class_masks[idx, iy * STEP: iy * STEP + PART_SIZE, ix * STEP: ix * STEP + PART_SIZE]
                                if save_statistics:
                                    self.classes2quantity[_cls] += 1
                                    self.classes2area[_cls] += np.sum(class_part)
                                part_has_class.append(_cls)
                                class_value = classes.index(_cls) + 1
                                mask[class_part] = class_value

                        if len(part_has_class) != 0:
                            os.makedirs(os.path.join(specters_output_path, folder_name), exist_ok=True)
                            os.makedirs(os.path.join(masks_output_path, folder_name), exist_ok=True)
                            np.save(os.path.join(specters_output_path, folder_name, f'{crop_counter}.npy'), part)
                            cv2.imwrite(os.path.join(masks_output_path, folder_name, f'{crop_counter}.png'), mask)

                            meta_info = {}
                            meta_info['original_filename'] = folder_name
                            meta_info['height'] = PART_SIZE
                            meta_info['width'] = PART_SIZE
                            meta_info['layersCount'] = part.shape[0]
                            meta_info['top_left'] = [iy * STEP, ix * STEP]
                            meta_info['classes'] = part_has_class
                            # meta_info['classes_distribution_images'] = self.classes2quantity
                            # meta_info['classes_distribution_pixels'] = self.classes2area

                            with open(os.path.join(specters_output_path, folder_name, f'{crop_counter}.yml'), 'w') as f:
                                yaml.dump(meta_info, f)
                            crop_counter += 1
