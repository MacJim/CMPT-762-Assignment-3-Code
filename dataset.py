import os
import typing
import json

import constant


# MARK: - JSON handling
# Keys: https://github.com/facebookresearch/detectron2/blob/master/docs/tutorials/datasets.md
ANNOTATION_ID_KEY: typing.Final = "id"    ; """"""
IMAGE_ID_KEY: typing.Final = "image_id"
SEGMENTATION_PATH_KEY: typing.Final = "segmentation"
CATEGORY_ID_KEY: typing.Final = "category_id"
CATEGORY_NAME_KEY: typing.Final = "category_name"
IS_CROWD_KEY: typing.Final = "iscrowd"
AREA_KEY: typing.Final = "area"
B_BOX_KEY: typing.Final = "bbox"
FILENAME_KEY: typing.Final = "file_name";    """Image filename. Example: 'P0000.png'."""


def _read_json_contents(filename: str) -> typing.List[typing.Dict[str, typing.Any]]:
    with open(filename, "r") as f:
        contents = json.load(f)

    return contents


# MARK: - List files
def _is_image(filename: str) -> bool:
    for extension in constant.IMAGE_FILE_EXTENSIONS:
        if (filename.endswith(extension)):
            return True

    return False


def _get_image_filenames(image_dir: str) -> typing.List[str]:
    filenames = os.listdir(image_dir)
    filenames = [f for f in filenames if _is_image(f)]
    filenames = [os.path.join(image_dir, f) for f in filenames]
    return filenames


# MARK: - Dataset
def get_detection_data(set_name: typing.Literal["train", "val", "test"]):
    """
    This function should return a list of data samples in which each sample is a dictionary.
    Make sure to select the correct bbox_mode for the data
    For the test data, you only have access to the images, therefore, the annotations should be empty.
    Other values could be obtained from the image files.

    :param set_name: "train", "val", "test"
    :return:
    """
    # TODO: approx 35 lines

    return_value = []

    return return_value
