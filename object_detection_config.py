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
from detectron2.data.transforms.augmentation import Augmentation
import detectron2.data.transforms as T
import  detectron2.structures
from PIL import Image
import numpy as np

import constant.detectron
import helper.bounding_box
import helper.segmentation_path
from helper.bounding_box import get_iou_xyxy, crop_bounding_box_xywh
from helper.segmentation_path import fit_segmentation_path_in_crop_box


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

    final_filename = os.path.join(output_dir, constant.detectron.FINAL_CHECKPOINT_FILENAME)
    if (os.path.isfile(final_filename)):
        print(f"Using final checkpoint file `{final_filename}`.")
        return final_filename
    else:
        alternative_files = os.listdir(output_dir)
        alternative_files = [f for f in alternative_files if f.endswith(".pth")]
        if not alternative_files:
            raise FileNotFoundError(f"No checkpoint file found in output dir `{output_dir}`!")

        alternative_files.sort()
        alternative_filename = os.path.join(output_dir, alternative_files[-1])
        print(f"Using alternative checkpoint file `{alternative_filename}`.")
        return alternative_filename


# MARK: - Baseline config
BASELINE_MODEL_YML: typing.Final = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
BASELINE_OUTPUT_DIR: typing.Final = "output"


def get_baseline_config(train=True) -> detectron2.config.config.CfgNode:
    """
    Get the baseline config in our project handout.

    Source: https://detectron2.readthedocs.io/en/latest/modules/config.html
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

    cfg.SOLVER.IMS_PER_BATCH = 2    # Batch size: images per batch
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


# MARK: - Naive config and dataloader
NAIVE_MODEL_YML: typing.Final = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"    # Not sure if I should use 152 layers.

NAIVE_OUTPUT_DIR: typing.Final = "output"    # Really don't want to change the default output dir.

NAIVE_CROP_PATCH_WIDTH: typing.Final = 800
NAIVE_CROP_PATCH_HEIGHT: typing.Final = 800
NAIVE_CROP_B_BOX_IOU_THRESHOLD: typing.Final = 0.6;    """Bounding boxes in cropped patch must have this much IoU with the original box."""


def get_naive_config(train=True) -> detectron2.config.config.CfgNode:
    if (train):
        _create_output_dir(NAIVE_OUTPUT_DIR)
    else:
        final_checkpoint_filename = _get_final_checkpoint_filename(NAIVE_OUTPUT_DIR)

    # Create configuration.
    cfg = detectron2.config.get_cfg()    # This is just a copy of the default config.

    cfg.merge_from_file(model_zoo.get_config_file(NAIVE_MODEL_YML))

    cfg.OUTPUT_DIR = NAIVE_OUTPUT_DIR

    cfg.DATASETS.TRAIN = (constant.detectron.TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (constant.detectron.TEST_DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.BACKBONE.FREEZE_AT = 2    # TODO: Do I freeze the first few stages?
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512    # Number of regions per image used to train RPN (Region Proposal Network)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1    # We only have a single plane class
    cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]    # ImageNet std
    if train:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(NAIVE_MODEL_YML)    # I don't know. Maybe loading a pre-trained model helps.
    else:
        cfg.MODEL.WEIGHTS = final_checkpoint_filename
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.STEPS = (500,)    # Learning rate scheduling
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.CHECKPOINT_PERIOD = 100    # Save a checkpoint after every this number of iterations.
    cfg.SOLVER.IMS_PER_BATCH = 3    # Batch size: images per batch

    return cfg


class NaiveTrainer (DefaultTrainer):
    """
    Custom trainer tutorials:

    - https://detectron2.readthedocs.io/en/latest/tutorials/training.html
    - https://detectron2.readthedocs.io/en/latest/modules/engine.html#detectron2.engine.defaults.DefaultTrainer
    - https://github.com/facebookresearch/detectron2/blob/master/projects/DeepLab/train_net.py
    """
    def __init__(self, cfg=None):
        if not cfg:
            cfg = get_naive_config(True)
        super(NaiveTrainer, self).__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        def naive_mapper(dataset_dict: typing.Dict[str, typing.Any]):
            """
            Calls the default mapper.

            This method is called "naive" due to historical reasons.
            It is now very complicated.

            :param dataset_dict:
            :return:
            """
            dataset_dict = copy.deepcopy(dataset_dict)    # We need to get rid of out of boundary bounding boxes, so we need copy here.

            # Get custom crop coordinates.
            class CustomCrop (Augmentation):
                """
                Refer to the implementation of `RandomCrop`: https://detectron2.readthedocs.io/en/latest/_modules/detectron2/data/transforms/augmentation_impl.html#RandomCrop
                """
                def __init__(self, x0: int, y0: int, crop_width: int, crop_height: int):
                    super(CustomCrop, self).__init__()
                    self._init(locals())

                def get_transform(self, image):
                    return T.CropTransform(self.x0, self.y0, self.crop_width, self.crop_height)

            # image_array: np.ndarray = detection_utils.read_image(dataset_dict["file_name"], format="BGR")    # HWC image
            image: Image.Image = Image.open(dataset_dict["file_name"])
            image_width, image_height = image.size
            del image

            max_x0 = image_width - NAIVE_CROP_PATCH_WIDTH
            if (max_x0 < 0):
                crop_x0 = 0
                crop_width = image_width
            else:
                crop_x0 = random.randint(0, max_x0)
                crop_width = NAIVE_CROP_PATCH_WIDTH

            max_y0 = image_height - NAIVE_CROP_PATCH_HEIGHT
            if (max_y0 < 0):
                crop_y0 = 0
                crop_height = image_height
            else:
                crop_y0 = random.randint(0, max_y0)
                crop_height = NAIVE_CROP_PATCH_HEIGHT

            # Get rid of bounding boxes that are out of boundary or cropped too much.
            annotations = dataset_dict[constant.detectron.ANNOTATIONS_KEY]
            retained_annotations = []
            for annotation in annotations:
                b_box = annotation[constant.detectron.B_BOX_KEY]    # xywh
                cropped_b_box = crop_bounding_box_xywh(b_box[0], b_box[1], b_box[2], b_box[3], crop_x0, crop_y0, crop_width, crop_height, NAIVE_CROP_B_BOX_IOU_THRESHOLD)
                if (cropped_b_box):
                    retained_annotations.append(annotation)

            dataset_dict[constant.detectron.ANNOTATIONS_KEY] = retained_annotations

            # Call the default mapper to crop both the image and the bounding boxes.
            mapper = DatasetMapper(cfg, is_train=True, augmentations=[
                # T.RandomCrop("absolute", (crop_height, crop_width)),
                CustomCrop(crop_x0, crop_y0, crop_width, crop_height),
                T.RandomBrightness(0.9, 1.1),
                T.RandomFlip(horizontal=True, vertical=False),
                T.RandomFlip(horizontal=False, vertical=True),    # Must do horizontal and vertical flips separately.
            ])
            return_value = mapper(dataset_dict)

            return return_value

        data_loader = build_detection_train_loader(cfg, mapper=naive_mapper)
        return data_loader


# MARK: - Mask R-CNN config and dataloader
MASK_R_CNN_MODEL_YML: typing.Final = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

MASK_R_CNN_OUTPUT_DIR: typing.Final = "mask_r_cnn_output"    # Want to make this different from the naive output dir.


def get_mask_r_cnn_config(train=True) -> detectron2.config.config.CfgNode:
    if (train):
        _create_output_dir(MASK_R_CNN_OUTPUT_DIR)
    else:
        final_checkpoint_filename = _get_final_checkpoint_filename(MASK_R_CNN_OUTPUT_DIR)

    # Create configuration.
    cfg = detectron2.config.get_cfg()  # This is just a copy of the default config.

    cfg.merge_from_file(model_zoo.get_config_file(MASK_R_CNN_MODEL_YML))

    cfg.OUTPUT_DIR = MASK_R_CNN_OUTPUT_DIR

    cfg.DATASETS.TRAIN = (constant.detectron.TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (constant.detectron.TEST_DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.BACKBONE.FREEZE_AT = 2    # TODO: Do I freeze the first few stages?
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512    # Number of regions per image used to train RPN (Region Proposal Network)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1    # We only have a single plane class
    cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]    # ImageNet std
    if train:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MASK_R_CNN_MODEL_YML)    # I don't know. Maybe loading a pre-trained model helps.
    else:
        cfg.MODEL.WEIGHTS = final_checkpoint_filename
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.STEPS = (200,)    # Learning rate scheduling
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.CHECKPOINT_PERIOD = 100    # Save a checkpoint after every this number of iterations.
    cfg.SOLVER.IMS_PER_BATCH = 3    # Batch size: images per batch

    return cfg


class MaskRCNNTrainer (NaiveTrainer):
    def __init__(self):
        super(MaskRCNNTrainer, self).__init__(get_mask_r_cnn_config(True))
