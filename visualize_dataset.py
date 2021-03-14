"""
Visualize 3 random samples of the training data using `detectron2.utils.visualizer`.

Refer to the official tutorial: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=UkNbUzUOLYf0&line=1&uniqifier=1
"""

import os
import random

import cv2
import detectron2.data
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
import torch

import dataset
import constant.detectron
from helper.visualization import save_visualization
from object_detection_config import get_baseline_config, get_naive_config, NaiveTrainer


def visualize_raw_dataset():
    dataset.register_datasets()

    train_dataset = detectron2.data.DatasetCatalog.get(constant.detectron.TRAIN_DATASET_NAME)
    train_metadata = detectron2.data.MetadataCatalog.get(constant.detectron.TRAIN_DATASET_NAME)
    for info_dict in random.sample(train_dataset, 3):
        image = cv2.imread(info_dict[constant.detectron.FILENAME_KEY])
        visualizer = Visualizer(image[:, :, ::-1], metadata=train_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(info_dict)
        out_image = out.get_image()[:, :, ::-1]

        base_filename = os.path.basename(info_dict[constant.detectron.FILENAME_KEY])
        out_filename = os.path.join("/tmp/visualization", base_filename)
        save_visualization(out_filename, out_image)


def visualize_data_loader():
    dataset.register_datasets()

    # cfg = get_baseline_config()
    # train_dataloader = DefaultTrainer.build_train_loader(cfg)

    cfg = get_naive_config()
    train_dataloader = NaiveTrainer.build_train_loader(cfg)

    for i, batch in enumerate(train_dataloader):
        if (i > 21):
            break

        for info_dict in batch:
            image: torch.Tensor = info_dict["image"]
            visualizer = Visualizer(image.numpy()[::-1, :, :].transpose(1, 2, 0), scale=0.5)
            # out = visualizer.draw_dataset_dict(info_dict)
            if ("instances" in info_dict):
                boxes = info_dict["instances"].get("gt_boxes")
                for box in boxes:
                    # print(box)
                    out = visualizer.draw_box(box)

                if not boxes:
                    out = visualizer.draw_dataset_dict(info_dict)    # I think this is NOP because the info dictionary returned by the data loader is different from the ones returned by the dataset.

            else:
                out = visualizer.draw_dataset_dict(info_dict)    # I think this is NOP because the info dictionary returned by the data loader is different from the ones returned by the dataset.

            out_image = out.get_image()[:, :, ::-1]

            base_filename = os.path.basename(info_dict[constant.detectron.FILENAME_KEY])
            out_filename = os.path.join("/tmp/visualization", base_filename)
            save_visualization(out_filename, out_image)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    # visualize_raw_dataset()
    visualize_data_loader()
