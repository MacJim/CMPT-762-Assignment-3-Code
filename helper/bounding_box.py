"""
Bounding box helpers.
"""


# MARK: - IoU
def get_iou(x01, y01, w1, h1, x02, y02, w2, h2):
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


def crop_bounding_boxes():
    pass
