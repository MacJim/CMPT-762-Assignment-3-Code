import os
import unittest
import random

import torch
from torch.utils import data

from segmentation_dataset import get_crop_coordinates, PlaneDataset
import dataset
import constant.dataset_file
from detectron2.data import DatasetCatalog


class CroppingTestCase (unittest.TestCase):
    def test_crop_coordinates(self):
        self.assertEqual(get_crop_coordinates(0, 100, 400, 200, padding_percentage=0.2), (-80, 20, 480, 380))
        self.assertEqual(get_crop_coordinates(0, 0, 400, 400, padding_percentage=0.2), (-80, -80, 480, 480))


class PlaneDatasetTestCase (unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset.register_datasets()

    def setUp(self) -> None:
        self.train_info_dicts = dataset.get_detection_data("train")
        self.test_info_dicts = dataset.get_detection_data("test")

    def test_item_type_from_dataset(self):
        width = 572
        height = 572

        train_dataset = PlaneDataset("train", self.train_info_dicts, width, height)
        for image, mask in train_dataset:
            with self.subTest():
                self.assertIsInstance(image, torch.Tensor)
                self.assertIsInstance(mask, torch.Tensor)

    def test_item_type_from_data_loader(self):
        batch_size = 6
        width = 572
        height = 572

        train_dataset = PlaneDataset("train", self.train_info_dicts, width, height)
        train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for image, mask in train_data_loader:
            with self.subTest():
                self.assertIsInstance(image, torch.Tensor)
                self.assertIsInstance(mask, torch.Tensor)

    def test_item_dimensions(self):
        batch_size = 6
        height = random.randint(80, 120)
        width = random.randint(80, 120)

        train_dataset = PlaneDataset("train", self.train_info_dicts, width, height)
        train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

        for image, mask in train_data_loader:
            with self.subTest():
                self.assertEqual(len(image.shape), 4)
                self.assertEqual(len(mask.shape), 3)

                self.assertEqual(image.shape[0], batch_size)
                self.assertEqual(mask.shape[0], batch_size)

                self.assertEqual(image.shape[1], 3)    # RGB

                self.assertEqual(image.shape[2], height)
                self.assertEqual(mask.shape[1], height)

                self.assertEqual(image.shape[3], width)
                self.assertEqual(mask.shape[2], width)

    def test_mask_values(self):
        """
        Mask only has 0 and 1 values.
        """
        batch_size = 6
        height = random.randint(80, 120)
        width = random.randint(80, 120)

        train_dataset = PlaneDataset("train", self.train_info_dicts, width, height)
        train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6)    # Data loader can't keep up with the GPU.

        for _, mask in train_data_loader:
            mask = mask.cuda()
            with self.subTest():
                self.assertEqual(torch.max(mask), 1.)
                self.assertEqual(torch.min(mask), 0.)

                mask[mask == 0.] = 1.
                self.assertTrue(torch.all(torch.eq(torch.square(mask), mask)))    # 1^2 == 1


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    unittest.main()
