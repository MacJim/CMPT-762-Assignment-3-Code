import os

from detectron2.engine import DefaultTrainer

import dataset
from object_detection_config import get_baseline_config


# MARK: - Main
def main():
    # Register datasets.
    dataset.register_datasets()

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
