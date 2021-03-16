import os
import typing

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

import segmentation_unet
from dataset import register_datasets, get_detection_data
from segmentation_dataset import PlaneDataset


# MARK: - Constants
N_CLASSES: typing.Final = 1
INPUT_WIDTH: typing.Final = 256
INPUT_HEIGHT: typing.Final = 256
CHECKPOINT_FILENAME: typing.Final = "../762A3-Checkpoints/seg_output_with_input_normalization/200.pth"


# MARK: - Network
network = segmentation_unet.NestedUNet(N_CLASSES)
network = network.cuda()
network.load_state_dict(torch.load(CHECKPOINT_FILENAME))
network.eval()


# MARK: - Training & validation data
VISUALIZATION_SAVE_DIR: typing.Final = "/tmp/seg_inference"


def visualize_training_patches():
    """
    Visualize 10 training/validation patches.
    """
    if (not os.path.exists(VISUALIZATION_SAVE_DIR)):
        os.makedirs(VISUALIZATION_SAVE_DIR)

    register_datasets()
    train_and_val_info_dicts = get_detection_data("train")
    train_and_val_dataset = PlaneDataset("train", train_and_val_info_dicts, INPUT_WIDTH, INPUT_HEIGHT, preload_images=False)
    data_loader = data.DataLoader(train_and_val_dataset, shuffle=True)

    for i, (image, mask) in enumerate(data_loader):
        if (i > 9):
            break

        image_cuda = image.cuda()

        with torch.no_grad():
            prediction = network(image_cuda)
            prediction = torch.sigmoid(prediction)

        # Remove batch dimension.
        image = torch.squeeze(image, 0)
        mask = torch.squeeze(mask, 0)
        prediction = torch.squeeze(prediction, 0)

        # Change mask and prediction values.
        mask *= 255.0
        prediction = torch.sigmoid(prediction)
        prediction *= 255.0

        save_image(image, os.path.join(VISUALIZATION_SAVE_DIR, f"{i}-image.png"))
        save_image(mask, os.path.join(VISUALIZATION_SAVE_DIR, f"{i}-mask.png"))
        save_image(prediction, os.path.join(VISUALIZATION_SAVE_DIR, f"{i}-prediction.png"))


def calculate_mean_iou():
    """
    Calculate training/validation mean IoU.
    """
    register_datasets()
    train_and_val_info_dicts = get_detection_data("train")
    train_and_val_dataset = PlaneDataset("train", train_and_val_info_dicts, INPUT_WIDTH, INPUT_HEIGHT, preload_images=False)
    # Don't use data loader workers when `preload_images=False`.
    # No I don't batch here because I want to check the IoU for each patch.
    data_loader = data.DataLoader(train_and_val_dataset, shuffle=False)

    ious = []

    for image, mask in data_loader:
        image = image.cuda()
        mask = mask.cuda()

        with torch.no_grad():
            prediction = network(image)
            prediction = torch.sigmoid(prediction)

        # Remove batch dimension.
        # image = torch.squeeze(image, 0)
        mask = torch.squeeze(mask, 0)
        prediction = torch.squeeze(prediction, 0)

        intersection = (prediction * mask).sum()
        union = prediction.sum() + mask.sum() - intersection
        iou = intersection / union
        print(f"Current IoU: {iou}")
        ious.append(iou.item())

    print(f"IoU: count: {len(ious)}, total: {sum(ious)}, mean: {sum(ious) / len(ious)}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    # visualize_training_patches()
    calculate_mean_iou()
