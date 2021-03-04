import os
# import argparse

import detectron2.config
from detectron2 import model_zoo

import constant.detectron


# MARK: - Constants
MODEL_YML = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
OUTPUT_DIR = "checkpoints"


# MARK: - Config
def update_config():
    """
    Source: https://detectron2.readthedocs.io/en/latest/modules/config.html
    """
    cfg = detectron2.config.get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(MODEL_YML))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_YML)

    cfg.DATASETS.TRAIN = (constant.detectron.TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (constant.detectron.TEST_DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 4    # Batch size: images per batch
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 400
    cfg.SOLVER.STEPS = []    # TODO: Learning rate scheduling

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256    # Number of regions per image used to train RPN (Region Proposal Network)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1    # We only have a single plane class

    cfg.OUTPUT_DIR = OUTPUT_DIR


# MARK: - Main
def main():
    # 1. Verify parameters

    # 2. Initialize config
    update_config()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_yml", type=str, default="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    # parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--learning_rate", type=float, default=0.00025)
    # parser.add_argument("--n_epochs", type=int, default=400)
    # parser.add_argument("--roi_per_image", type=int, default=256, help="ROIs per image to train the Region Proposal Network. (default: %(default)s)")
    # parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output dir. (default: %(default)s)")
    # parser.add_argument("--train_log_filename", type=str, default="checkpoints/train_log.csv")
    # parser.add_argument("--validation_log_filename", type=str, default="checkpoints/validation_log.csv")
    # args = parser.parse_args()

    main()
