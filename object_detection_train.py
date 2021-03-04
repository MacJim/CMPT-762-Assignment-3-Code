import os
# import argparse

import detectron2.config
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer

import constant.detectron


def _create_output_dir(dir_name: str):
    if (os.path.exists(dir_name)):
        if (os.path.isdir(dir_name)):
            print(f"Using existing output dir {dir_name}")
            print(f"Files in existing output dir may be overwritten.")
        else:
            raise FileExistsError(f"Output dir {dir_name} is a file or link!")
    else:
        os.makedirs(dir_name)
        print(f"Created output dir {dir_name}")


# MARK: - Config
def get_baseline_config() -> detectron2.config.config.CfgNode:
    """
    Get the baseline config in our project handout.

    Source: https://detectron2.readthedocs.io/en/latest/modules/config.html
    """
    model_yml = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    output_dir = "output"

    _create_output_dir(output_dir)

    cfg = detectron2.config.get_cfg()    # This is just a copy of the default config.

    cfg.merge_from_file(model_zoo.get_config_file(model_yml))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yml)    # I don't think I should load the weights in the baseline.

    cfg.DATASETS.TRAIN = (constant.detectron.TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (constant.detectron.TEST_DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 2    # Batch size: images per batch
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 500
    # cfg.SOLVER.STEPS = []    # Learning rate scheduling: none in the baseline

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512    # Number of regions per image used to train RPN (Region Proposal Network)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1    # We only have a single plane class

    cfg.OUTPUT_DIR = output_dir

    return cfg


# MARK: - Main
def main():
    # Get config.
    cfg = get_baseline_config()

    # Train.
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)    # Start from iteration 0.
    trainer.train()    # Run training.


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
