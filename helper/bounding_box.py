"""
Bounding box helpers.
"""

import typing


# MARK: - IoU
def get_iou_xywh(x01, y01, w1, h1, x02, y02, w2, h2):
    x11 = x01 + w1
    y11 = y01 + h1
    x12 = x02 + w2
    y12 = y02 + h2

    intersection_x0 = max(x01, x02)
    intersection_y0 = max(y01, y02)
    intersection_x1 = min(x11, x12)
    intersection_y1 = min(y11, y12)

    if ((intersection_x0 >= intersection_x1) or (intersection_y0 >= intersection_y1)):
        return 0.0

    width = intersection_x1 - intersection_x0
    height = intersection_y1 - intersection_y0
    intersection = width * height
    union = w1 * h1 + w2 * h2 - intersection
    return (intersection / union)


def crop_bounding_box_xywh(bounding_box_x0, bounding_box_y0, bounding_box_w, bounding_box_h, crop_box_x0, crop_box_y0, crop_box_w, crop_box_h, threshold: float) -> typing.Optional[typing.Tuple[int, int, int, int]]:
    if (get_iou_xywh(bounding_box_x0, bounding_box_y0, bounding_box_w, bounding_box_h, crop_box_x0, crop_box_y0, crop_box_w, crop_box_h) == 0.0):
        return None

    bounding_box_x1 = bounding_box_x0 + bounding_box_w
    bounding_box_y1 = bounding_box_y0 + bounding_box_h
    crop_box_x1 = crop_box_x0 + crop_box_w
    crop_box_y1 = crop_box_y0 + crop_box_h

    new_x0 = max(bounding_box_x0, crop_box_x0)
    new_x1 = min(bounding_box_x1, crop_box_x1)
    new_y0 = max(bounding_box_y0, crop_box_y0)
    new_y1 = min(bounding_box_y1, crop_box_y1)
    new_w = new_x1 - new_x0
    new_h = new_y1 - new_y0

    if (get_iou_xywh(new_x0, new_y0, new_w, new_h, bounding_box_x0, bounding_box_y0, bounding_box_w, bounding_box_h) < threshold):
        # The bounding box was cropped too much.
        return None

    return (new_x0, new_y0, new_w, new_h)
