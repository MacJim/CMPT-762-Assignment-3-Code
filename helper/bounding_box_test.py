import os
import unittest

from bounding_box import get_iou_xywh, crop_bounding_box_xywh, get_containing_bounding_box_xyxy


class IoUTestCase (unittest.TestCase):
    def test_zero(self):
        # No intersection on y
        self.assertLess(get_iou_xywh(0., 0., 1., 1., 1.2, 0.5, 1., 1.), 0.01)
        self.assertLess(get_iou_xywh(1.2, 0.5, 1., 1., 0., 0., 1., 1.), 0.01)

        # No intersection on y
        self.assertLess(get_iou_xywh(0., 0., 1., 1., 0.5, 1.2, 1., 1.), 0.01)
        self.assertLess(get_iou_xywh(0.5, 1.2, 1., 1., 0., 0., 1., 1.), 0.01)

        # No intersection on x and y
        self.assertLess(get_iou_xywh(0., 0., 1., 1., 1.2, 1.2, 1., 1.), 0.01)
        self.assertLess(get_iou_xywh(1.2, 1.2, 1., 1., 0., 0., 1., 1.), 0.01)

    def test_include(self):
        """
        One area includes another.
        """
        self.assertEqual(get_iou_xywh(0., 0., 2., 2., 0.5, 0.5, 1., 1.), 0.25)
        self.assertEqual(get_iou_xywh(0.5, 0.5, 1., 1., 0., 0., 2., 2.), 0.25)

        self.assertEqual(get_iou_xywh(0., 0., 2., 2., 1., 0., 1., 2.), 0.5)
        self.assertEqual(get_iou_xywh(1., 0., 1., 2., 0., 0., 2., 2.), 0.5)

        self.assertAlmostEqual(get_iou_xywh(0., 0., 2., 2., 0., 0.5, 2., 1.), 0.5)
        self.assertAlmostEqual(get_iou_xywh(0., 0.5, 2., 1., 0., 0., 2., 2.), 0.5)

    def test_intersect(self):
        self.assertAlmostEqual(get_iou_xywh(0., 0., 2., 2., 1., 1., 2., 2.), 1 / 7)
        self.assertAlmostEqual(get_iou_xywh(1., 1., 2., 2., 0., 0., 2., 2.), 1 / 7)

        self.assertAlmostEqual(get_iou_xywh(0., 0., 2., 2., 0., 1., 2., 2.), 1 / 3)
        self.assertAlmostEqual(get_iou_xywh(0., 1., 2., 2., 0., 0., 2., 2.), 1 / 3)


class BoundingBoxCroppingTestCase (unittest.TestCase):
    def test_none(self):
        threshold = 0.5

        # No intersection on y
        self.assertIsNone(crop_bounding_box_xywh(0., 0., 1., 1., 1.2, 0.5, 1., 1., threshold))
        self.assertIsNone(crop_bounding_box_xywh(1.2, 0.5, 1., 1., 0., 0., 1., 1., threshold))

        # No intersection on y
        self.assertIsNone(crop_bounding_box_xywh(0., 0., 1., 1., 0.5, 1.2, 1., 1., threshold))
        self.assertIsNone(crop_bounding_box_xywh(0.5, 1.2, 1., 1., 0., 0., 1., 1., threshold))

        # No intersection on x and y
        self.assertIsNone(crop_bounding_box_xywh(0., 0., 1., 1., 1.2, 1.2, 1., 1., threshold))
        self.assertIsNone(crop_bounding_box_xywh(1.2, 1.2, 1., 1., 0., 0., 1., 1., threshold))

    def test_include(self):
        threshold = 0.5

        self.assertEqual(crop_bounding_box_xywh(1., 1., 1., 1., 0., 0., 3., 3., threshold), (1., 1., 1., 1.))

    def test_intersect(self):
        threshold = 0.5

        self.assertIsNone(crop_bounding_box_xywh(0., 0., 2., 2., 1., 1., 3., 3., threshold))
        self.assertEqual(crop_bounding_box_xywh(0., 1., 4., 3., 1., 1., 3., 3., threshold), (0., 0., 3., 3.))

        self.assertIsNone(crop_bounding_box_xywh(2., 2., 3., 3., 1., 1., 3., 3., threshold))
        self.assertEqual(crop_bounding_box_xywh(2., 2., 2., 3., 1., 1., 3., 3., threshold), (1., 1., 2., 2.))

        self.assertEqual(crop_bounding_box_xywh(2., 0., 2., 3., 1., 1., 3., 3., threshold), (1., 0., 2., 2.))


class ContainingBoxTestCase (unittest.TestCase):
    def test_non_intersection(self):
        self.assertEqual(get_containing_bounding_box_xyxy([[0, 0, 2, 2], [3, 3, 6, 6]]), [0, 0, 6, 6])
        self.assertEqual(get_containing_bounding_box_xyxy([[3, 3, 6, 6], [0, 0, 2, 2]]), [0, 0, 6, 6])
        self.assertEqual(get_containing_bounding_box_xyxy([[3, 3, 6, 6], [0, 0, 2, 2], [7, 7, 9, 9]]), [0, 0, 9, 9])

    def test_intersection(self):
        self.assertEqual(get_containing_bounding_box_xyxy([[0, 0, 2, 2], [1, 1, 3, 3]]), [0, 0, 3, 3])
        self.assertEqual(get_containing_bounding_box_xyxy([[1, 1, 3, 3], [0, 0, 2, 2]]), [0, 0, 3, 3])

        self.assertEqual(get_containing_bounding_box_xyxy([[0, 0, 2, 2], [1, 1, 3, 3], [0, 2, 2, 4]]), [0, 0, 3, 4])
        self.assertEqual(get_containing_bounding_box_xyxy([[0, 0, 2, 2], [0, 2, 2, 4], [1, 1, 3, 3]]), [0, 0, 3, 4])

    def test_include(self):
        self.assertEqual(get_containing_bounding_box_xyxy([[0, 0, 4, 4], [1, 1, 2, 2], [2, 2, 4, 4]]), [0, 0, 4, 4])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    unittest.main()
