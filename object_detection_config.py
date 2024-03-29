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
    def __init__(self):
        super(NaiveTrainer, self).__init__(get_naive_config(True))

    @classmethod
    def build_train_loader(cls, cfg):
        def naive_mapper(dataset_dict: typing.Dict[str, typing.Any]):
            """
            Naively calls the default mapper.

            TODO: This solution has a major flaw: segmentation boxes may be cut and make the object in it incomplete.

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

            # # Identify bounding boxes that are cropped too much.
            # instances = return_value["instances"]    # Instances(num_instances=2, image_height=800, image_width=800, fields=[gt_boxes: Boxes(tensor([[  0.,  33., 236., 241.], [  0., 722., 242., 800.]])), gt_classes: tensor([0, 0])])
            # patch_boxes: torch.Tensor = instances.get("gt_boxes").tensor
            #
            # new_patch_boxes_list = []
            # for i in range(patch_boxes.shape[0]):
            #     patch_box = patch_boxes[i].tolist()    # Format: x0, y0, x1, y1
            #     restored_patch_box = [patch_box[0] + crop_x0, patch_box[1] + crop_y0, patch_box[2] + crop_x0, patch_box[3] + crop_y0]    # Format: x0, y0, x1, y1
            #     max_iou = 0.0
            #     for annotation in dataset_dict[constant.detectron.ANNOTATIONS_KEY]:
            #         if constant.detectron.B_BOX_KEY not in annotation:
            #             continue
            #
            #         original_b_box = annotation[constant.detectron.B_BOX_KEY]    # Format: x0, y0, w, h
            #         iou = get_iou_xyxy(restored_patch_box[0], restored_patch_box[1], restored_patch_box[2], restored_patch_box[3], original_b_box[0], original_b_box[1], original_b_box[0] + original_b_box[2], original_b_box[1] + original_b_box[3])
            #         max_iou = max(iou, max_iou)
            #
            #     if (max_iou >= NAIVE_CROP_B_BOX_IOU_THRESHOLD):
            #         new_patch_boxes_list.append(patch_box)
            #
            # if new_patch_boxes_list:
            #     new_patch_boxes = torch.tensor(new_patch_boxes_list)
            #     new_gt_classes = torch.zeros((len(new_patch_boxes_list)))    # We only have one class, thus we directly use 0 here.
            # else:
            #     # We still need the secondary shape.
            #     new_patch_boxes = torch.zeros((0, 4))
            #     new_gt_classes = torch.zeros((0,))
            #
            # # Create a new `Instances` object based on the retained bounding boxes.
            # new_instances = detectron2.structures.Instances((crop_height, crop_width))
            # new_instances.set("gt_boxes", detectron2.structures.Boxes(new_patch_boxes))
            # new_instances.set("gt_classes", new_gt_classes)
            #
            # return_value["instances"] = new_instances
            #
            # return return_value

        data_loader = build_detection_train_loader(cfg, mapper=naive_mapper)
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
            max_x0 = width - NAIVE_CROP_PATCH_WIDTH
            max_y0 = height - NAIVE_CROP_PATCH_HEIGHT

            if (max_x0 >= 0):
                x0 = random.randint(0, max_x0)
                x1 = x0 + NAIVE_CROP_PATCH_WIDTH
            else:
                x0 = 0
                x1 = width - 1

            if (max_y0 >= 0):
                y0 = random.randint(0, max_y0)
                y1 = y0 + NAIVE_CROP_PATCH_HEIGHT
            else:
                y0 = 0
                y1 = height - 1

            image: Image.Image = image.crop((x0, y0, x1, y1))

            # Crop bounding boxes and segmentations.
            cropped_annotations = []
            for annotation in dataset_dict[constant.detectron.ANNOTATIONS_KEY]:
                b_box = annotation[constant.detectron.B_BOX_KEY]
                cropped_b_box = helper.bounding_box.crop_bounding_box_xywh(b_box[0], b_box[1], b_box[2], b_box[3], x0, y0, x1 - x0, y1 - y0, NAIVE_CROP_B_BOX_IOU_THRESHOLD)
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
