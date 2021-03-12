import os
import unittest
import random

from segmentation_path import fit_segmentation_path_in_crop_box


class SegmentationPathCroppingTestCase (unittest.TestCase):
    def test_inside(self):
        path = [random.randint(20, 80) for _ in range(100000)]
        result = fit_segmentation_path_in_crop_box(path, 20, 20, 60, 60)
        expected_result = [(value - 20) for value in path]
        self.assertEqual(result, expected_result)

    def test_outside(self):
        path = [random.randint(0, 100) for _ in range(100000)]
        result = fit_segmentation_path_in_crop_box(path, 20, 20, 60, 60)

        expected_result = []
        for value in path:
            if (value < 20):
                expected_result.append(0)
            elif (value > 80):
                expected_result.append(80)
            else:
                expected_result.append(value - 20)

        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    unittest.main()
