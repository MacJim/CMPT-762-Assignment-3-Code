import os
import typing

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

import segmentation_unet
from dataset import register_datasets, get_detection_data
from segmentation_dataset import PlaneDataset


N_CLASSES: typing.Final = 1
INPUT_WIDTH: typing.Final = 256
INPUT_HEIGHT: typing.Final = 256
CHECKPOINT_FILENAME: typing.Final = "seg_output/140.pth"
SAVE_DIR: typing.Final = "/tmp/seg_inference"


def main():
    if (not os.path.exists(SAVE_DIR)):
        os.makedirs(SAVE_DIR)

    network = segmentation_unet.NestedUNet(N_CLASSES)
    network = network.cuda()
    network.load_state_dict(torch.load(CHECKPOINT_FILENAME))
    network.eval()

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

        # Remove batch dimension.
        image = torch.squeeze(image, 0)
        mask = torch.squeeze(mask, 0)
        prediction = torch.squeeze(prediction, 0)

        # Change mask and prediction values.
        mask *= 255.0
        prediction = torch.sigmoid(prediction)
        prediction *= 255.0

        save_image(image, os.path.join(SAVE_DIR, f"{i}-image.png"))
        save_image(mask, os.path.join(SAVE_DIR, f"{i}-mask.png"))
        save_image(prediction, os.path.join(SAVE_DIR, f"{i}-prediction.png"))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    main()
