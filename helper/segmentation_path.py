"""
Segmentation path helpers.
"""

import typing
import copy


def fit_segmentation_path_in_crop_box(path: typing.List[int], crop_box_x0, crop_box_y0, crop_box_w, crop_box_h, set_outside_values_to_boundary_value=True):
    return_value = copy.copy(path)

    crop_box_x1 = crop_box_x0 + crop_box_w
    crop_box_y1 = crop_box_y0 + crop_box_h

    def get_cropped_x_value(x: int) -> int:
        if (set_outside_values_to_boundary_value):
            if (x < crop_box_x0):
                return 0
            elif (x > crop_box_x1):
                return crop_box_x1

        return (x - crop_box_x0)

    def get_cropped_y_value(y: int) -> int:
        if (set_outside_values_to_boundary_value):
            if (y < crop_box_y0):
                return 0
            elif (y > crop_box_y1):
                return crop_box_y1

        return (y - crop_box_y0)

    for i in range(0, len(return_value), 2):
        return_value[i] = get_cropped_x_value(return_value[i])

    for j in range(1, len(return_value), 2):
        return_value[j] = get_cropped_y_value(return_value[j])

    return return_value
