import os
import typing
import json

from PIL import Image
import detectron2.data

import constant.dataset_file
import constant.detectron


# MARK: - JSON handling
def _read_json_contents(filename: str) -> typing.List[typing.Dict[str, typing.Any]]:
    with open(filename, "r") as f:
        contents = json.load(f)

    return contents


# MARK: - List files
def _is_image(filename: str) -> bool:
    for extension in constant.dataset_file.IMAGE_FILE_EXTENSIONS:
        if (filename.endswith(extension)):
            return True

    return False


def _get_image_filenames(image_dir: str) -> typing.List[str]:
    filenames = os.listdir(image_dir)
    filenames = [f for f in filenames if _is_image(f)]
    filenames = [os.path.join(image_dir, f) for f in filenames]
    return filenames


# MARK: - Dataset
def _add_json_entry_annotations_to_info_dict(json_entry: typing.Dict[str, typing.Any], info_dict: typing.Dict[str, typing.Any]):
    """
    Adds the information in a JSON entry to the info dictionary.

    :param json_entry:
    :param info_dict:
    :return: None
    """
    if constant.detectron.ANNOTATIONS_KEY not in info_dict:
        info_dict[constant.detectron.ANNOTATIONS_KEY] = []

    annotation = {
        constant.detectron.B_BOX_KEY: json_entry[constant.dataset_file.B_BOX_KEY],
        constant.detectron.B_BOX_MODE_KEY: constant.detectron.DESIGNATED_BOX_MODE,
        constant.detectron.SEGMENTATION_PATH_KEY: json_entry[constant.detectron.SEGMENTATION_PATH_KEY],
        constant.detectron.CATEGORY_ID_KEY: constant.dataset_file.CATEGORY_ID_KEY,
    }

    info_dict[constant.detectron.ANNOTATIONS_KEY].append(annotation)


def get_detection_data(set_name: typing.Literal["train", "val", "test"]):
    """
    This function should return a list of data samples in which each sample is a dictionary.
    Make sure to select the correct bbox_mode for the data
    For the test data, you only have access to the images, therefore, the annotations should be empty.
    Other values could be obtained from the image files.

    :param set_name: "train", "val", "test"
    :return:
    """
    # Get filenames
    if (set_name == "test"):
        image_dir = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TEST_DATASET_SUB_DIR)
        json_contents = []

    else:
        image_dir = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TRAIN_DATASET_SUB_DIR)

        train_json_filename = os.path.join(constant.dataset_file.DATASET_ROOT_DIR, constant.dataset_file.TRAIN_JSON_FILENAME)
        json_contents = _read_json_contents(train_json_filename)

    image_filenames = _get_image_filenames(image_dir)

    return_value = [{constant.detectron.FILENAME_KEY: f} for f in image_filenames]

    # Add file height and width info by reading the files.
    for info_dict in return_value:
        with Image.open(info_dict[constant.detectron.FILENAME_KEY]) as image:    # Auto close
            info_dict[constant.detectron.WIDTH_KEY], info_dict[constant.detectron.HEIGHT_KEY] = image.size    # PIL is width first

    if json_contents:
        # Train data.
        # Add image ID and annotations.
        for json_entry in json_contents:
            filename_json = json_entry[constant.dataset_file.FILENAME_KEY]

            for info_dict in return_value:
                filename_info_dict = os.path.basename(info_dict[constant.detectron.FILENAME_KEY])
                if (filename_json == filename_info_dict):
                    # if ((constant.detectron.IMAGE_ID_KEY in info_dict) and (info_dict[constant.detectron.IMAGE_ID_KEY] != json_entry[constant.dataset_file.IMAGE_ID_KEY])):
                    #     print("Contradiction!")    # No contradictions. But image IDs seem quite random and don't start at 0.
                    info_dict[constant.detectron.IMAGE_ID_KEY] = json_entry[constant.dataset_file.IMAGE_ID_KEY]
                    _add_json_entry_annotations_to_info_dict(json_entry, info_dict)

    else:
        # Test data.
        # Add automatic image ID.
        for i, info_dict in enumerate(return_value):
            info_dict[constant.detectron.IMAGE_ID_KEY] = i

    return return_value


def register_datasets():
    """
    Register the train and test datasets.
    """
    detectron2.data.DatasetCatalog.register(constant.detectron.TRAIN_DATASET_NAME, lambda set_name="train": get_detection_data(set_name))
    detectron2.data.MetadataCatalog.get(constant.detectron.TRAIN_DATASET_NAME).things_classes = [constant.dataset_file.DESIGNATED_CATEGORY_NAME]

    detectron2.data.DatasetCatalog.register(constant.detectron.TEST_DATASET_NAME, lambda set_name="test": get_detection_data(set_name))
    detectron2.data.MetadataCatalog.get(constant.detectron.TEST_DATASET_NAME).things_classes = [constant.dataset_file.DESIGNATED_CATEGORY_NAME]


if __name__ == '__main__':
    get_detection_data("train")
    # get_detection_data("test")
