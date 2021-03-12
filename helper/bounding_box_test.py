import os
import unittest

from bounding_box import get_iou


class IoUTestCase (unittest.TestCase):
    def test_zero(self):
        # No intersection on y
        self.assertLess(get_iou(0., 0., 1., 1., 1.2, 0.5, 1., 1.), 0.01)
        self.assertLess(get_iou(1.2, 0.5, 1., 1., 0., 0., 1., 1.), 0.01)

        # No intersection on y
        self.assertLess(get_iou(0., 0., 1., 1., 0.5, 1.2, 1., 1.), 0.01)
        self.assertLess(get_iou(0.5, 1.2, 1., 1., 0., 0., 1., 1.), 0.01)

        # No intersection on x and y
        self.assertLess(get_iou(0., 0., 1., 1., 1.2, 1.2, 1., 1.), 0.01)
        self.assertLess(get_iou(1.2, 1.2, 1., 1., 0., 0., 1., 1.), 0.01)

    def test_include(self):
        """
        One area includes another.
        """
        self.assertEqual(get_iou(0., 0., 2., 2., 0.5, 0.5, 1., 1.), 0.25)
        self.assertEqual(get_iou(0.5, 0.5, 1., 1., 0., 0., 2., 2.), 0.25)

        self.assertEqual(get_iou(0., 0., 2., 2., 1., 0., 1., 2.), 0.5)
        self.assertEqual(get_iou(1., 0., 1., 2., 0., 0., 2., 2.), 0.5)

        self.assertAlmostEqual(get_iou(0., 0., 2., 2., 0., 0.5, 2., 1.), 0.5)
        self.assertAlmostEqual(get_iou(0., 0.5, 2., 1., 0., 0., 2., 2.), 0.5)

    def test_intersection(self):
        self.assertAlmostEqual(get_iou(0., 0., 2., 2., 1., 1., 2., 2.), 1 / 7)
        self.assertAlmostEqual(get_iou(1., 1., 2., 2., 0., 0., 2., 2.), 1 / 7)

        self.assertAlmostEqual(get_iou(0., 0., 2., 2., 0., 1., 2., 2.), 1 / 3)
        self.assertAlmostEqual(get_iou(0., 1., 2., 2., 0., 0., 2., 2.), 1 / 3)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    unittest.main()
