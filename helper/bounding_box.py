"""
Bounding box helpers.
"""

import typing
import warnings


# MARK: - IoU
def get_iou_xyxy(x01, y01, x11, y11, x02, y02, x12, y12) -> float:
    w1 = x11 - x01
    h1 = y11 - y01
    w2 = x12 - x02
    h2 = y12 - y02

    return_value = get_iou_xywh(x01, y01, w1, h1, x02, y02, w2, h2)
    return return_value


def get_iou_xywh(x01, y01, w1, h1, x02, y02, w2, h2) -> float:
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


# MARK: - Crop
def crop_bounding_box_xywh(bounding_box_x0, bounding_box_y0, bounding_box_w, bounding_box_h, crop_box_x0, crop_box_y0, crop_box_w, crop_box_h, threshold: float) -> typing.Optional[typing.Tuple[int, int, int, int]]:
    """

    :param bounding_box_x0:
    :param bounding_box_y0:
    :param bounding_box_w:
    :param bounding_box_h:
    :param crop_box_x0:
    :param crop_box_y0:
    :param crop_box_w:
    :param crop_box_h:
    :param threshold:
    :return: (x0, y0, w, h) in the crop box's coordinate.
    """
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

    # Convert to crop box's coordinate.
    new_x0 -= crop_box_x0
    new_y0 -= crop_box_y0
    del new_x1
    del new_y1

    return (new_x0, new_y0, new_w, new_h)


# MARK: - Contains/containing box
def contains_box_xyxy(box0: typing.List[float], box1: typing.List[float]) -> int:
    """
    Judge if `box0` and `box1` contain each other.

    :param box0: (x0, y0, x1, y1)
    :param box1: (x0, y0, x1, y1)
    :return: 0 if they don't contain each other
    """
    warnings.warn("Not implemented")
    pass


def get_containing_bounding_box_xyxy(sub_boxes: typing.List[typing.List[float]]) -> typing.List[float]:
    """
    Get a large bounding box that contains the given bounding boxes.

    :param sub_boxes: [(x0, y0, x1, y1), ...]
    :return: (x0, y0, x1, y1)
    """
    if not sub_boxes:
        raise ValueError("Sub boxes must not be empty!")

    x0 = sub_boxes[0][0]
    y0 = sub_boxes[0][1]
    x1 = sub_boxes[0][2]
    y1 = sub_boxes[0][3]

    for box in sub_boxes:
        x0 = min(x0, box[0])
        y0 = min(y0, box[1])
        x1 = max(x1, box[2])
        y1 = max(y1, box[3])

    return [x0, y0, x1, y1]


# MARK: - NMS
def nms_xyxy(pred_boxes: typing.List[typing.List[float]], pred_scores: typing.List[float], nms_iou_threshold=0.5) -> typing.Tuple[typing.List[typing.List[float]], typing.List[float]]:
    # Sort according to scores: largest score first, smallest score last.
    pred_scores, pred_boxes = (list(t) for t in zip(*sorted(zip(pred_scores, pred_boxes), reverse=True)))

    returned_scores = []
    returned_boxes = []

    for score, box in zip(pred_scores, pred_boxes):
        if not returned_scores:
            returned_scores.append(score)
            returned_boxes.append(box)
            continue

        keep_current_box = True
        for existing_box in returned_boxes:
            iou = get_iou_xyxy(box[0], box[1], box[2], box[3], existing_box[0], existing_box[1], existing_box[2], existing_box[3])
            if (iou > nms_iou_threshold):
                keep_current_box = False
                break

        if keep_current_box:
            returned_scores.append(score)
            returned_boxes.append(box)

    return (returned_boxes, returned_scores)


# MARK: - Combine
def combine_bounding_boxes_naively_xyxy(pred_boxes: typing.List[typing.List[float]], nms_iou_threshold=0.8):
    """
    Combine a set of possibly duplicate bounding boxes into 1.

    This function should be used along with the naive trainer.

    For the moment, the strategy is very naive:

    1. Find all boxes covered by other boxes, and remove them
    2. NMS among all the remaining boxes
    """
    # 1
    retained_boxes = []
    for box in pred_boxes:
        intersection_found = False

        for existing_box in retained_boxes:
            # Check if current box is contained in another box, or vice versa.
            intersection_x0 = max(box[0], existing_box[0])
            intersection_y0 = max(box[1], existing_box[1])
            intersection_x1 = min(box[2], existing_box[2])
            intersection_y1 = min(box[3], existing_box[3])

            if ((intersection_x0 >= intersection_x1) or (intersection_y0 >= intersection_y1)):
                # No intersection.
                continue

            if ((get_iou_xyxy(intersection_x0, intersection_y0, intersection_x1, intersection_y1, box[0], box[1], box[2], box[3]) > nms_iou_threshold) or (get_iou_xyxy(intersection_x0, intersection_y0, intersection_x1, intersection_y1, existing_box[0], existing_box[1], existing_box[2], existing_box[3]))):
                # One box's mostly inside another box.
                existing_box[0] = min(box[0], existing_box[0])
                existing_box[1] = min(box[1], existing_box[1])
                existing_box[2] = max(box[2], existing_box[2])
                existing_box[3] = max(box[3], existing_box[3])
                intersection_found = True
                break

        if not intersection_found:
            retained_boxes.append(box)

    return retained_boxes
