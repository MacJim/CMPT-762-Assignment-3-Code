import os
import unittest

from patch import get_crop_patch_axes


class GetCropPatchesTestCase (unittest.TestCase):
    def test_crop_patch_0_axis(self):
        self.assertEqual(get_crop_patch_axes(100, 100, 800, 800, 200), [[0, 100, 0, 100]])
        self.assertEqual(get_crop_patch_axes(800, 800, 800, 800, 200), [[0, 800, 0, 800]])

    def test_crop_patch_1_axis(self):
        self.assertEqual(get_crop_patch_axes(100, 1300, 800, 800, 200), [[0, 100, 0, 800], [0, 100, 200, 1000], [0, 100, 400, 1200], [0, 100, 500, 1300]])
        self.assertEqual(get_crop_patch_axes(100, 1000, 800, 800, 200), [[0, 100, 0, 800], [0, 100, 200, 1000]])
        self.assertEqual(get_crop_patch_axes(100, 999, 800, 800, 200), [[0, 100, 0, 800], [0, 100, 199, 999]])

        self.assertEqual(get_crop_patch_axes(1300, 100, 800, 800, 200), [[0, 800, 0, 100], [200, 1000, 0, 100], [400, 1200, 0, 100], [500, 1300, 0, 100]])
        self.assertEqual(get_crop_patch_axes(1000, 100, 800, 800, 200), [[0, 800, 0, 100], [200, 1000, 0, 100]])
        self.assertEqual(get_crop_patch_axes(999, 100, 800, 800, 200), [[0, 800, 0, 100], [199, 999, 0, 100]])

    def test_crop_patch_2_axes(self):
        self.assertEqual(get_crop_patch_axes(999, 999, 800, 800, 200), [[0, 800, 0, 800], [199, 999, 0, 800], [0, 800, 199, 999], [199, 999, 199, 999]])
        self.assertEqual(get_crop_patch_axes(1300, 1300, 800, 800, 200),[
            [0, 800, 0, 800],
            [200, 1000, 0, 800],
            [400, 1200, 0, 800],
            [500, 1300, 0, 800],

            [0, 800, 200, 1000],
            [200, 1000, 200, 1000],
            [400, 1200, 200, 1000],
            [500, 1300, 200, 1000],

            [0, 800, 400, 1200],
            [200, 1000, 400, 1200],
            [400, 1200, 400, 1200],
            [500, 1300, 400, 1200],

            [0, 800, 500, 1300],
            [200, 1000, 500, 1300],
            [400, 1200, 500, 1300],
            [500, 1300, 500, 1300],
        ])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    unittest.main()
