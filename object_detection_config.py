"""
Get Detectron configuration (`detectron2.config.config.CfgNode`).
"""

import os
import typing

import detectron2.config
from detectron2 import model_zoo

import constant.detectron


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
