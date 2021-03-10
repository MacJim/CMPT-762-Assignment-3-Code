import os
import unittest
from collections import defaultdict

import detectron2.data
import numpy as np
import matplotlib.pyplot as plt

import dataset
import constant.dataset_file
import constant.detectron


class JsonTestCase (unittest.TestCase):
    def test_json_type(self):
        json_filename = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        self.assertIsNotNone(json_contents)
        self.assertIsInstance(json_contents, list)
        for item in json_contents:
            with self.subTest(item=item):
                self.assertIsInstance(item, dict)

    def test_json_keys(self):
        json_filename = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        for item in json_contents:
            with self.subTest(item=item):
                self.assertEqual(len(item), 9)    # All items have this many key value pairs

                self.assertIsInstance(item[constant.dataset_file.ANNOTATION_ID_KEY], int)
                self.assertIsInstance(item[constant.dataset_file.IMAGE_ID_KEY], int)
                self.assertIsInstance(item[constant.dataset_file.SEGMENTATION_PATH_KEY], list)
                self.assertIsInstance(item[constant.dataset_file.CATEGORY_ID_KEY], int)
                self.assertIsInstance(item[constant.dataset_file.CATEGORY_NAME_KEY], str)
                self.assertIsInstance(item[constant.dataset_file.IS_CROWD_KEY], int)
                self.assertIsInstance(item[constant.dataset_file.AREA_KEY], int)
                self.assertIsInstance(item[constant.dataset_file.B_BOX_KEY], list)
                self.assertIsInstance(item[constant.dataset_file.FILENAME_KEY], str)

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
        json_filename = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        for item in json_contents:
            with self.subTest(item=item):
                path_list = item[constant.dataset_file.SEGMENTATION_PATH_KEY]

                self.assertIsInstance(path_list, list)
                # self.assertEqual(len(path_list), 1)
                self.assertGreater(len(path_list), 0)

                path = path_list[0]
                self.assertIsInstance(path, list)
                self.assertEqual(len(path) % 2, 0)    # (x0, y0, x1, y1, ...) Length must be dividable by 2.

    def test_bounding_boxes(self):
        json_filename = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        for item in json_contents:
            with self.subTest(item=item):
                b_boxes = item[constant.dataset_file.B_BOX_KEY]
                self.assertIsInstance(b_boxes, list)
                self.assertEqual(len(b_boxes), 4)

    def test_multiple_segmentation_paths(self):
        """
        Most images have more than 1 segmentation paths.
        """
        json_filename = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        image_filenames_and_segmentation_paths = defaultdict(int)

        for item in json_contents:
            path_list = item[constant.dataset_file.SEGMENTATION_PATH_KEY]
            for _ in path_list:
                image_filenames_and_segmentation_paths[item[constant.dataset_file.FILENAME_KEY]] += 1

        self.assertGreater(len(image_filenames_and_segmentation_paths), 0)

        for filename, path_count in image_filenames_and_segmentation_paths.items():
            with self.subTest(filename=filename, path_count=path_count):
                self.assertGreaterEqual(path_count, 1)

    def test_train_image_file_existence(self):
        json_filename = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TRAIN_JSON_FILENAME)
        json_contents = dataset._read_json_contents(json_filename)

        train_image_dir = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TRAIN_DATASET_SUB_DIR)

        for item in json_contents:
            with self.subTest(item=item):
                image_filename = item[constant.dataset_file.FILENAME_KEY]
                image_filename = os.path.join(train_image_dir, image_filename)
                self.assertTrue(os.path.isfile(image_filename))


class FileTestCase (unittest.TestCase):
    def test_is_image(self):
        image_filenames = ["2.png", "P2106.png", "2.jpg", "P2106.jpg"]
        non_image_filenames = ["2", "P2106.pn", "2.ini", "P2106.jpg2", ".DS_Store"]

        for filename in image_filenames:
            with self.subTest(filename=filename):
                self.assertTrue(dataset._is_image(filename))

        for filename in non_image_filenames:
            with self.subTest(filename=filename):
                self.assertFalse(dataset._is_image(filename))

    def test_file_existence(self):
        train_image_dir = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TRAIN_DATASET_SUB_DIR)
        train_filenames = dataset._get_image_filenames(train_image_dir)
        self.assertEqual(len(train_filenames), 198)
        for filename in train_filenames:
            with self.subTest(filename=filename):
                self.assertTrue(os.path.isfile(filename))

        test_image_dir = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TEST_DATASET_SUB_DIR)
        test_filenames = dataset._get_image_filenames(test_image_dir)
        self.assertEqual(len(test_filenames), 72)
        for filename in test_filenames:
            with self.subTest(filename=filename):
                self.assertTrue(os.path.isfile(filename))


class DatasetTestCase (unittest.TestCase):
    def setUp(self) -> None:
        super(DatasetTestCase, self).setUp()

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        self.train_info_dicts = dataset.get_detection_data("train")
        self.test_info_dicts = dataset.get_detection_data("test")
        
    def tearDown(self) -> None:
        super(DatasetTestCase, self).tearDown()

    def test_lengths(self):
        self.assertEqual(len(self.train_info_dicts), 198)
        self.assertEqual(len(self.test_info_dicts), 72)

    def test_train_file_existence(self):
        for info_dict in self.train_info_dicts:
            filename = info_dict[constant.detectron.FILENAME_KEY]
            with self.subTest(filename=filename):
                self.assertTrue(os.path.isfile(filename))

    def test_test_file_existence(self):
        for info_dict in self.test_info_dicts:
            filename = info_dict[constant.detectron.FILENAME_KEY]
            with self.subTest(filename=filename):
                self.assertTrue(os.path.isfile(filename))

    def test_train_image_ids(self):
        """
        Train images do have unique IDs.
        """
        existing_image_ids = []
        for info_dict in self.train_info_dicts:
            image_id = info_dict[constant.detectron.IMAGE_ID_KEY]
            with self.subTest(image_id=image_id):
                if image_id not in existing_image_ids:
                    existing_image_ids.append(image_id)
                else:
                    self.fail(f"{image_id} already exists!")

    def test_test_image_ids(self):
        for i, info_dict in enumerate(self.test_info_dicts):
            image_id = info_dict[constant.detectron.IMAGE_ID_KEY]
            with self.subTest(i=i, image_id=image_id):
                self.assertEqual(i, image_id)

    def test_train_image_keys(self):
        for info_dict in self.train_info_dicts:
            with self.subTest():
                self.assertIn(constant.detectron.FILENAME_KEY, info_dict)
                self.assertIn(constant.detectron.IMAGE_ID_KEY, info_dict)
                self.assertIn(constant.detectron.HEIGHT_KEY, info_dict)
                self.assertIn(constant.detectron.WIDTH_KEY, info_dict)

                self.assertIn(constant.detectron.ANNOTATIONS_KEY, info_dict)
                for annotation in info_dict[constant.detectron.ANNOTATIONS_KEY]:
                    with self.subTest():
                        self.assertIn(constant.detectron.B_BOX_KEY, annotation)
                        self.assertIn(constant.detectron.B_BOX_MODE_KEY, annotation)
                        self.assertIn(constant.detectron.SEGMENTATION_PATH_KEY, annotation)
                        self.assertIn(constant.detectron.CATEGORY_ID_KEY, annotation)

    def test_test_image_keys(self):
        """
        Test images have no annotations.
        """
        for info_dict in self.test_info_dicts:
            with self.subTest():
                self.assertIn(constant.detectron.FILENAME_KEY, info_dict)
                self.assertIn(constant.detectron.IMAGE_ID_KEY, info_dict)
                self.assertIn(constant.detectron.HEIGHT_KEY, info_dict)
                self.assertIn(constant.detectron.WIDTH_KEY, info_dict)

                self.assertNotIn(constant.detectron.ANNOTATIONS_KEY, info_dict)

    def test_train_bounding_box_attributes(self):
        """
        Find the min/max width, min
        """
        widths = []
        heights = []
        sizes = []
        aspect_ratios = []

        for info_dict in self.train_info_dicts:
            annotations = info_dict[constant.detectron.ANNOTATIONS_KEY]

            for annotation in annotations:
                with self.subTest():
                    b_box_mode = annotation[constant.detectron.B_BOX_MODE_KEY]
                    self.assertEqual(b_box_mode, constant.detectron.DESIGNATED_BOX_MODE)

                    b_box = annotation[constant.detectron.B_BOX_KEY]
                    width = b_box[2]
                    height = b_box[3]

                    widths.append(width)
                    heights.append(height)
                    sizes.append(width * height)
                    aspect_ratios.append(width / height)

        width_height_bin_edges = [0, 10, 25, 50, 100, 200, 400, 800, 1200, 10000]
        size_bin_edges = [0, 10, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 100000000]
        aspect_ratio_bin_edges = [(i / 10) for i in range(0, 11)] + [(10 / i) for i in reversed(range(1, 10))] + [1000000]

        plt.figure(figsize=(15.0, 9.0))    # width, height

        hist, _ = np.histogram(widths, width_height_bin_edges)
        plt.barh(range(len(hist)), hist, tick_label=[f"{width_height_bin_edges[i]} ~ {width_height_bin_edges[i + 1]}" for i, _ in enumerate(hist)])
        plt.gca().set(title=f"Widths: max: {max(widths)}, min: {min(widths)}, median: {np.median(widths)}, average: {np.average(widths):.3f}")
        plt.savefig("/tmp/widths.png")
        plt.clf()

        hist, _ = np.histogram(heights, width_height_bin_edges)
        plt.barh(range(len(hist)), hist, tick_label=[f"{width_height_bin_edges[i]} ~ {width_height_bin_edges[i + 1]}" for i, _ in enumerate(hist)])
        plt.gca().set(title=f"Heights: max: {max(heights)}, min: {min(heights)}, median: {np.median(heights)}, average: {np.average(heights):.3f}")
        plt.savefig("/tmp/heights.png")
        plt.clf()

        hist, _ = np.histogram(sizes, size_bin_edges)
        plt.barh(range(len(hist)), hist, tick_label=[f"{size_bin_edges[i]} ~ {size_bin_edges[i + 1]}" for i, _ in enumerate(hist)])
        plt.gca().set(title=f"Sizes: max: {max(sizes)}, min: {min(sizes)}, median: {np.median(sizes)}, average: {np.average(sizes):.3f}")
        plt.savefig("/tmp/sizes.png")
        plt.clf()

        hist, _ = np.histogram(aspect_ratios, aspect_ratio_bin_edges)
        plt.barh(range(len(hist)), hist, tick_label=[f"{aspect_ratio_bin_edges[i]:.3f} ~ {aspect_ratio_bin_edges[i + 1]:.3f}" for i, _ in enumerate(hist)])
        plt.gca().set(title=f"Aspect ratios (width/height): max: {max(aspect_ratios):.3f}, min: {min(aspect_ratios):.3f}, median: {np.median(aspect_ratios):.3f}, average: {np.average(aspect_ratios):.3f}")
        plt.savefig("/tmp/aspect_ratios.png")
        plt.clf()


class DatasetRegistrationTestCase (unittest.TestCase):
    def setUp(self) -> None:
        super(DatasetRegistrationTestCase, self).setUp()

        dataset.register_datasets()

    def test_lengths(self):
        train_dataset = detectron2.data.DatasetCatalog.get(constant.detectron.TRAIN_DATASET_NAME)
        test_dataset = detectron2.data.DatasetCatalog.get(constant.detectron.TEST_DATASET_NAME)

        self.assertEqual(len(train_dataset), 198)
        self.assertEqual(len(test_dataset), 72)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    unittest.main()
