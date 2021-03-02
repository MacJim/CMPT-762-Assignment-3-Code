import typing

from detectron2.structures import BoxMode


DATASET_ROOT_DIR: typing.Final = "data/"

TRAIN_DATASET_SUB_DIR: typing.Final = "train/"
TEST_DATASET_SUB_DIR: typing.Final = "test/"
IMAGE_FILE_EXTENSIONS: typing.Final = [".png", ".jpg"]

TRAIN_JSON_FILENAME: typing.Final = "train.json"
DESIGNATED_CATEGORY_ID: typing.Final = 4
DESIGNATED_CATEGORY_NAME: typing.Final = "plane"

# Keys: https://github.com/facebookresearch/detectron2/blob/master/docs/tutorials/datasets.md
ANNOTATION_ID_KEY: typing.Final = "id"
IMAGE_ID_KEY: typing.Final = "image_id"
SEGMENTATION_PATH_KEY: typing.Final = "segmentation"
CATEGORY_ID_KEY: typing.Final = "category_id"
CATEGORY_NAME_KEY: typing.Final = "category_name"
IS_CROWD_KEY: typing.Final = "iscrowd"
AREA_KEY: typing.Final = "area"
B_BOX_KEY: typing.Final = "bbox"
FILENAME_KEY: typing.Final = "file_name";    """Image filename. Example: 'P0000.png'."""

BOX_MODE: typing.Final = BoxMode.XYWH_ABS
"""
https://detectron2.readthedocs.io/en/latest/_modules/detectron2/structures/boxes.html

`XYWH_REL` means range [0, 1] and is not yet supported.
"""
