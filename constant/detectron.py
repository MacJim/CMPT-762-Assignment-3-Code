"""
Detectron constants.

Source: https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
"""

import typing

from detectron2.structures import BoxMode


# MARK: Detectron keys
# The following keys are to be obtained from the image file and added to our return value.
FILENAME_KEY: typing.Final = "file_name"
IMAGE_ID_KEY: typing.Final = "image_id"
HEIGHT_KEY: typing.Final = "height";    """Height in integer."""
WIDTH_KEY: typing.Final = "width";    """Width in integer."""
ANNOTATIONS_KEY: typing.Final = "annotations"    ; """Annotations: list[dict]"""
B_BOX_KEY: typing.Final = "bbox"
B_BOX_MODE_KEY: typing.Final = "bbox_mode"
CATEGORY_ID_KEY: typing.Final = "category_id"    ; """An integer in the range [0, num_categories-1] representing the category label. The value num_categories is reserved to represent the “background” category, if applicable."""
SEGMENTATION_PATH_KEY: typing.Final = "segmentation"    ; """list[list[float]]"""
IS_CROWD_KEY: typing.Final = "iscrowd"    ; """0 (default) or 1. Whether this instance is labeled as COCO’s “crowd region”."""

# Designated values.
DESIGNATED_BOX_MODE: typing.Final = BoxMode.XYWH_ABS
"""
https://detectron2.readthedocs.io/en/latest/_modules/detectron2/structures/boxes.html

`XYWH_REL` means range [0, 1] and is not yet supported.
"""


# MARK: Detectron dataset names
TRAIN_DATASET_NAME: typing.Final = "data_detection_train"
TEST_DATASET_NAME: typing.Final = "data_detection_test"
