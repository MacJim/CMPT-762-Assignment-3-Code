import os
import unittest
from collections import defaultdict

import dataset
import constant


class DatasetTestCase (unittest.TestCase):
    def test_json_type(self):
        json_filename = os.path.join(constant.DATASET_ROOT_DIR, constant.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        self.assertIsNotNone(json_contents)
        self.assertIsInstance(json_contents, list)
        for item in json_contents:
            with self.subTest(item=item):
                self.assertIsInstance(item, dict)

    def test_json_keys(self):
        json_filename = os.path.join(constant.DATASET_ROOT_DIR, constant.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        for item in json_contents:
            with self.subTest(item=item):
                self.assertEqual(len(item), 9)    # All items have this many key value pairs

                self.assertIsInstance(item[dataset.ANNOTATION_ID_KEY], int)
                self.assertIsInstance(item[dataset.IMAGE_ID_KEY], int)
                self.assertIsInstance(item[dataset.SEGMENTATION_PATH_KEY], list)
                self.assertIsInstance(item[dataset.CATEGORY_ID_KEY], int)
                self.assertIsInstance(item[dataset.CATEGORY_NAME_KEY], str)
                self.assertIsInstance(item[dataset.IS_CROWD_KEY], int)
                self.assertIsInstance(item[dataset.AREA_KEY], int)
                self.assertIsInstance(item[dataset.B_BOX_KEY], list)
                self.assertIsInstance(item[dataset.FILENAME_KEY], str)

    def test_segmentation_paths(self):
        """
        Most contain a single segmentation path.
        11 of them contain 2 or 3 segmentation paths:

        - 'id': 24565, 'image_id': 81
        - 'id': 29355, 'image_id': 124
        - 'id': 78089, 'image_id': 571
        - 'id': 78188, 'image_id': 571
        - 'id': 78703, 'image_id': 579
        - 'id': 94525, 'image_id': 590
        - 'id': 170725, 'image_id': 923
        - 'id': 170789, 'image_id': 923
        - 'id': 205140, 'image_id': 1104
        - 'id': 209795, 'image_id': 1125
        - 'id': 355960, 'image_id': 1409

        Since each entry only contains a single bounding box, that should mean: some bounding boxes correspond to multiple segmentation paths.
        Or it could mean: each entry is an instance.
        """
        json_filename = os.path.join(constant.DATASET_ROOT_DIR, constant.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        for item in json_contents:
            with self.subTest(item=item):
                path_list = item[dataset.SEGMENTATION_PATH_KEY]

                self.assertIsInstance(path_list, list)
                # self.assertEqual(len(path_list), 1)
                self.assertGreater(len(path_list), 0)

                path = path_list[0]
                self.assertIsInstance(path, list)
                self.assertEqual(len(path) % 2, 0)    # (x0, y0, x1, y1, ...) Length must be dividable by 2.

    def test_bounding_boxes(self):
        json_filename = os.path.join(constant.DATASET_ROOT_DIR, constant.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        for item in json_contents:
            with self.subTest(item=item):
                b_boxes = item[dataset.B_BOX_KEY]
                self.assertIsInstance(b_boxes, list)
                self.assertEqual(len(b_boxes), 4)

    def test_multiple_segmentation_paths(self):
        """
        Most images have more than 1 segmentation paths.
        """
        json_filename = os.path.join(constant.DATASET_ROOT_DIR, constant.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        image_filenames_and_segmentation_paths = defaultdict(int)

        for item in json_contents:
            path_list = item[dataset.SEGMENTATION_PATH_KEY]
            for _ in path_list:
                image_filenames_and_segmentation_paths[item[dataset.FILENAME_KEY]] += 1

        self.assertGreater(len(image_filenames_and_segmentation_paths), 0)

        for filename, path_count in image_filenames_and_segmentation_paths.items():
            with self.subTest(filename=filename, path_count=path_count):
                self.assertGreaterEqual(path_count, 1)

    def test_train_image_file_existence(self):
        json_filename = os.path.join(constant.DATASET_ROOT_DIR, constant.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        train_image_dir = os.path.join(constant.DATASET_ROOT_DIR, constant.TRAIN_DATASET_SUB_DIR)

        for item in json_contents:
            with self.subTest(item=item):
                image_filename = item[dataset.FILENAME_KEY]
                image_filename = os.path.join(train_image_dir, image_filename)
                self.assertTrue(os.path.isfile(image_filename))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    unittest.main()
