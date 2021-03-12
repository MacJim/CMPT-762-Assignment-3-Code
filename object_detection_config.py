"""
Get Detectron configuration (`detectron2.config.config.CfgNode`).
"""

import os
import typing
import copy
import random

import torch
import torchvision.transforms.functional as TF
import detectron2.config
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader, detection_utils
import detectron2.data.transforms as T
from PIL import Image
import numpy as np

import constant.detectron
import helper.bounding_box
import helper.segmentation_path


# MARK: - Output dir manipulation
def _create_output_dir(dir_name: str):
    if (os.path.exists(dir_name)):
        if (os.path.isdir(dir_name)):
            print(f"Using existing output dir `{dir_name}`")
            print(f"Files in existing output dir may be overwritten.")
        else:
            raise FileExistsError(f"Output dir `{dir_name}` is a file or link!")
    else:
        os.makedirs(dir_name)
        print(f"Created output dir `{dir_name}`")


def _get_final_checkpoint_filename(output_dir: str) -> str:
    """
    :param output_dir: Output directory.
    :return: Path of the final checkpoint file.
    """
    if (not os.path.exists(output_dir)):
        raise FileNotFoundError(f"Output dir `{output_dir}` doesn't exist!")
    elif (not os.path.isdir(output_dir)):
        raise ValueError(f"Output dir `{output_dir}` exists, but is not a directory.")

    filename = os.path.join(output_dir, constant.detectron.FINAL_CHECKPOINT_FILENAME)
    if (os.path.isfile(filename)):
        print(f"Using checkpoint file `{filename}`")
        return filename
    elif (not os.path.exists(filename)):
        raise FileNotFoundError(f"Checkpoint file `{filename}` does not exist!")
    else:
        raise ValueError(f"Checkpoint file `{filename}` exists, but is not a file.")


# MARK: - Baseline config
BASELINE_MODEL_YML: typing.Final = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
BASELINE_OUTPUT_DIR: typing.Final = "output"


def get_baseline_config(train=True) -> detectron2.config.config.CfgNode:
    """
    Get the baseline config in our project handout.

    Source: https://detectron2.readthedocs.io/en/latest/modules/config.htmlx
    """
    if (train):
        # Train: create the output dir.
        _create_output_dir(BASELINE_OUTPUT_DIR)
    else:
        # Inference: the output dir must already exist.
        final_checkpoint_filename = _get_final_checkpoint_filename(BASELINE_OUTPUT_DIR)

    # Create configuration.
    cfg = detectron2.config.get_cfg()    # This is just a copy of the default config.

    cfg.merge_from_file(model_zoo.get_config_file(BASELINE_MODEL_YML))

    cfg.OUTPUT_DIR = BASELINE_OUTPUT_DIR

    cfg.DATASETS.TRAIN = (constant.detectron.TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (constant.detectron.TEST_DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 3    # Batch size: images per batch
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 500
    # cfg.SOLVER.STEPS = []    # Learning rate scheduling: none in the baseline

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512    # Number of regions per image used to train RPN (Region Proposal Network)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1    # We only have a single plane class
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yml)    # I don't think I should load the weights in the baseline.
    if (not train):
        cfg.MODEL.WEIGHTS = final_checkpoint_filename
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

    return cfg


# MARK: - Custom config and custom dataloader
CUSTOM_MODEL_YML: typing.Final = "COCO-Detection/faster_rcnn_X_101_FPN_3x.yaml"    # Not sure if I should use 152 layers.

CUSTOM_CONFIG_OUTPUT_DIR: typing.Final = "output"    # Really don't want to change the default output dir.

CUSTOM_CONFIG_CROP_PATCH_WIDTH: typing.Final = 800
CUSTOM_CONFIG_CROP_PATCH_HEIGHT: typing.Final = 800
CUSTOM_CROP_B_BOX_IOU_THRESHOLD: typing.Final = 0.5;    """Bounding boxes in cropped patch must have this much IoU with the original box."""

CUSTOM_CONFIG_AUGMENTATIONS: typing.Final = []
"""
TODO: https://detectron2.readthedocs.io/en/latest/tutorials/augmentation.html
"""


def get_custom_config(train=True) -> detectron2.config.config.CfgNode:
    return get_baseline_config()    # TODO: Remove

    if (train):
        _create_output_dir(CUSTOM_CONFIG_OUTPUT_DIR)
    else:
        final_checkpoint_filename = _get_final_checkpoint_filename(CUSTOM_CONFIG_OUTPUT_DIR)

    cfg = detectron2.config.get_cfg()  # This is just a copy of the default config.

    return cfg


class CustomTrainer (DefaultTrainer):
    """
    Custom trainer tutorials:

    - https://detectron2.readthedocs.io/en/latest/tutorials/training.html
    - https://detectron2.readthedocs.io/en/latest/modules/engine.html#detectron2.engine.defaults.DefaultTrainer
    - https://github.com/facebookresearch/detectron2/blob/master/projects/DeepLab/train_net.py
    """
    def __init__(self):
        super(CustomTrainer, self).__init__(get_custom_config(True))

    @classmethod
    def build_train_loader(cls, cfg):
        def custom_mapper(dataset_dict: typing.Dict[str, typing.Any]):
            """
            Naively calls the default mapper.

            TODO: This solution has a major flaw: segmentation boxes may be cut and make the object in it incomplete.

            :param dataset_dict:
            :return:
            """
            image_array: np.ndarray = detection_utils.read_image(dataset_dict["file_name"], format="BGR")    # HWC image

            max_y0 = image_array.shape[0] - CUSTOM_CONFIG_CROP_PATCH_HEIGHT
            if (max_y0 < 0):
                y0 = 0
                height = image_array.shape[0]
            else:
                y0 = random.randint(0, max_y0)
                height = CUSTOM_CONFIG_CROP_PATCH_HEIGHT

            max_x0 = image_array.shape[1] - CUSTOM_CONFIG_CROP_PATCH_WIDTH
            if (max_x0 < 0):
                x0 = 0
                width = image_array.shape[1]
            else:
                x0 = random.randint(0, max_x0)
                width = CUSTOM_CONFIG_CROP_PATCH_WIDTH

            # Just call the default mapper.
            mapper = DatasetMapper(cfg, is_train=True, augmentations=[
                T.RandomBrightness(0.9, 1.1),
                T.RandomCrop("absolute", (height, width)),
            ])
            return_value = mapper(dataset_dict)
            return return_value

            # augs = T.AugmentationList([
            #     T.RandomBrightness(0.9, 1.1),
            #     T.RandomCrop("absolute", (height, width)),
            # ])

            # aug_input = T.AugInput(image_array)
            # transform = augs(aug_input)
            #
            # image_tensor: torch.Tensor = torch.from_numpy(aug_input.image.transpose(2, 0, 1))
            #
            # annotations = [detection_utils.transform_instance_annotations(annotation, [transform], image_tensor.shape[1:]) for annotation in dataset_dict[constant.detectron.ANNOTATIONS_KEY]]
            # annotation_instances = detection_utils.annotations_to_instances(annotations, image_tensor.shape[1:])
            #
            # return {
            #     "image": image_tensor,
            #     "instances": annotation_instances,
            # }

        data_loader = build_detection_train_loader(cfg, mapper=custom_mapper)
        return data_loader

    @classmethod
    def build_train_loader_legacy(cls, cfg: detectron2.config.config.CfgNode) -> typing.Iterable:
        """
        https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html
        """
        def custom_mapper(dataset_dict: typing.Dict[str, typing.Any]):
            dataset_dict = copy.deepcopy(dataset_dict)    # Avoid contaminating the original dict
            filename = dataset_dict[constant.detectron.FILENAME_KEY]

            # Read image.
            # image = detection_utils.read_image(filename)    # np.ndarray, channel last, seems to be the original resolution, inconvenient
            image: Image.Image = Image.open(filename)
            width, height = image.size    # At this time, `width` equals `dataset_dict["width"]`, `height` equals `dataset_dict["height"]`
            # print(width, height, dataset_dict["width"], dataset_dict["height"])

            # Crop image.
            max_x0 = width - CUSTOM_CONFIG_CROP_PATCH_WIDTH
            max_y0 = height - CUSTOM_CONFIG_CROP_PATCH_HEIGHT

            if (max_x0 >= 0):
                x0 = random.randint(0, max_x0)
                x1 = x0 + CUSTOM_CONFIG_CROP_PATCH_WIDTH
            else:
                x0 = 0
                x1 = width - 1

            if (max_y0 >= 0):
                y0 = random.randint(0, max_y0)
                y1 = y0 + CUSTOM_CONFIG_CROP_PATCH_HEIGHT
            else:
                y0 = 0
                y1 = height - 1

            image: Image.Image = image.crop((x0, y0, x1, y1))

            # Crop bounding boxes and segmentations.
            cropped_annotations = []
            for annotation in dataset_dict[constant.detectron.ANNOTATIONS_KEY]:
                b_box = annotation[constant.detectron.B_BOX_KEY]
                cropped_b_box = helper.bounding_box.crop_bounding_box_xywh(b_box[0], b_box[1], b_box[2], b_box[3], x0, y0, x1 - x0, y1 - y0, CUSTOM_CROP_B_BOX_IOU_THRESHOLD)
                if not cropped_b_box:
                    continue

                segmentations = annotation[constant.detectron.SEGMENTATION_PATH_KEY]
                cropped_segmentations = [helper.segmentation_path.fit_segmentation_path_in_crop_box(s, x0, y0, x1 - x0, y1 - y0) for s in segmentations]

                cropped_annotation = copy.deepcopy(annotation)
                cropped_annotation[constant.detectron.B_BOX_KEY] = list(cropped_b_box)    # By default, `cropped_b_box` is a tuple.
                cropped_annotation[constant.detectron.SEGMENTATION_PATH_KEY] = cropped_segmentations
                cropped_annotations.append(cropped_annotation)

            # Convert to model input format.
            image_tensor: torch.Tensor = TF.to_tensor(image)
            annotation_instances = detection_utils.annotations_to_instances(cropped_annotations, (y1 - y0, x1 - x0))

            return {
                "image": image_tensor,
                "instances": annotation_instances,
                constant.detectron.FILENAME_KEY: filename,
                constant.detectron.WIDTH_KEY: (x1 - x0),
                constant.detectron.HEIGHT_KEY: (y1 - y0),
                constant.detectron.IMAGE_ID_KEY: dataset_dict[constant.detectron.IMAGE_ID_KEY],
            }

        data_loader = build_detection_train_loader(cfg, mapper=custom_mapper)
        return data_loader
