import os
import unittest

import dataset
from detectron2.engine import DefaultTrainer
from object_detection_config import get_baseline_config, get_custom_config, CustomTrainer

import constant.detectron


class DefaultTrainerTestCase (unittest.TestCase):
    def setUp(self) -> None:
        super(DefaultTrainerTestCase, self).setUp()

        dataset.register_datasets()

    def test_data_loader(self):
        """
        Model input format: https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format
        """
        cfg = get_baseline_config()
        data_loader = DefaultTrainer.build_train_loader(cfg)

        # Iteration count seem limitless.
        iteration_limit = 10
        for i, batch in enumerate(data_loader):
            if (i >= iteration_limit):
                break

            with self.subTest():
                print(f"{i}. Type: {type(batch)}, len: {len(batch)}")
                self.assertIsInstance(batch, list)
                self.assertEqual(len(batch), cfg.SOLVER.IMS_PER_BATCH)    # Batch size
                for info_dict in batch:
                    print(f"Filename: {info_dict[constant.detectron.FILENAME_KEY]}, instances: {info_dict['instances']}, shape: {info_dict['image'].shape}, dtype: {info_dict['image'].dtype}")
                    self.assertIsInstance(info_dict, dict)


class CustomTrainerTestCase (unittest.TestCase):
    def setUp(self) -> None:
        super(CustomTrainerTestCase, self).setUp()

        dataset.register_datasets()

    def test_data_loader(self):
        cfg = get_custom_config()
        data_loader = CustomTrainer.build_train_loader(cfg)

        iteration_limit = 10
        for i, batch in enumerate(data_loader):
            if (i >= iteration_limit):
                break

            with self.subTest():
                print(f"{i}. Type: {type(batch)}, len: {len(batch)}")
                self.assertIsInstance(batch, list)
                self.assertEqual(len(batch), cfg.SOLVER.IMS_PER_BATCH)  # Batch size
                for info_dict in batch:
                    print(f"Filename: {info_dict[constant.detectron.FILENAME_KEY]}, instances: {info_dict['instances']}, shape: {info_dict['image'].shape}, dtype: {info_dict['image'].dtype}")
                    self.assertIsInstance(info_dict, dict)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    unittest.main()
