import unittest

import icecream

from hsi_dataset_api import HsiDataset


class TestDataset(unittest.TestCase):
    def test(self):
        dataset = HsiDataset('../example/dataset_example')
        icecream.ic(dataset.get_dataset_description())
        for data_point in dataset.data_iterator(opened=False):
            icecream.ic(data_point)

        icecream.ic(dataset.data_readall_as_list(opened=False))

        icecream.ic(dataset.data_readall_as_list(opened=True, train=True))

