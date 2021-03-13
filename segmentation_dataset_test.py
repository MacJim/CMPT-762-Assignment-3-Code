import os
import unittest

from segmentation_dataset import get_crop_coordinates


class CroppingTest (unittest.TestCase):
    def test_crop_coordinates(self):
        self.assertEqual(get_crop_coordinates(0, 100, 400, 200, padding_percentage=0.2), (-80, 20, 480, 380))
        self.assertEqual(get_crop_coordinates(0, 0, 400, 400, padding_percentage=0.2), (-80, -80, 480, 480))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    unittest.main()
