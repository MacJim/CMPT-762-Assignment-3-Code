import typing
import json

import constant


# MARK: - JSON handling
ANNOTATION_ID_KEY = "id"    ; """"""
IMAGE_ID_KEY = "image_id"
SEGMENTATION_PATH_KEY = "segmentation"
CATEGORY_ID_KEY = "category_id"
CATEGORY_NAME_KEY = "category_name"
IS_CROWD_KEY = "iscrowd"
AREA_KEY = "area"
B_BOX_KEY = "bbox"
FILENAME_KEY = "file_name";    """Image filename. Example: 'P0000.png'."""


def _read_json_contents(filename: str) -> typing.List[typing.Dict[str, typing.Any]]:
    with open(filename, "r") as f:
        contents = json.load(f)

    return contents


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
