import os
import typing

import torch
from torch.utils import data
from torchvision.utils import save_image

from dataset import register_datasets, get_detection_data
from segmentation_dataset import PlaneDataset


INPUT_WIDTH: typing.Final = 256
INPUT_HEIGHT: typing.Final = 256
SAVE_DIR: typing.Final = "/tmp/seg_visualization"


def visualize_data_loader():
    if (not os.path.exists(SAVE_DIR)):
        os.makedirs(SAVE_DIR)

    register_datasets()
    train_and_val_info_dicts = get_detection_data("train")
    train_and_val_dataset = PlaneDataset("train", train_and_val_info_dicts, INPUT_WIDTH, INPUT_HEIGHT, preload_images=False)
    data_loader = data.DataLoader(train_and_val_dataset, shuffle=True)

    for i, (image, mask) in enumerate(data_loader):
        if (i > 9):
            break

        image = torch.squeeze(image, 0)
        mask = torch.squeeze(mask, 0)
        mask *= 255.0

        save_image(image, os.path.join(SAVE_DIR, f"{i}-image.png"))
        save_image(mask, os.path.join(SAVE_DIR, f"{i}-mask.png"))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    # visualize_raw_dataset()
    visualize_data_loader()
