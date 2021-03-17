import os
import typing

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image

import segmentation_unet
from dataset import register_datasets, get_detection_data
from segmentation_dataset import PlaneDataset, TRANSFORM_MEAN, TRANSFORM_STD


N_CLASSES: typing.Final = 1
INPUT_WIDTH: typing.Final = 256
INPUT_HEIGHT: typing.Final = 256
# CHECKPOINT_FILENAME: typing.Final = "../762A3-Checkpoints/seg_output_with_input_normalization/240.pth"
CHECKPOINT_FILENAME: typing.Final = "seg_output/240.pth"
SAVE_DIR: typing.Final = "/tmp/seg_inference"


network = segmentation_unet.NestedUNet(N_CLASSES)
network = network.cuda()
network.load_state_dict(torch.load(CHECKPOINT_FILENAME))
network.eval()


def infer_test_patch(filename: str, crop_x0, crop_y0, crop_x1, crop_y1):
    image: Image.Image = Image.open(filename)
    # Sometimes `image` is grayscale (mode "L").
    # We'll need to convert it to RGB.
    if (image.mode != "RGB"):
        image = image.convert("RGB")

    image: Image.Image = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    image: Image.Image = image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.ANTIALIAS)
    image_tensor: torch.Tensor = TF.to_tensor(image)
    image_tensor = TF.normalize(image_tensor, TRANSFORM_MEAN, TRANSFORM_STD)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    image_tensor = image_tensor.cuda()

    with torch.no_grad():
        prediction = network(image_tensor)

    # Remove batch dimension.
    prediction = torch.squeeze(prediction, 0)
    prediction = torch.sigmoid(prediction)

    return_value = torch.zeros(prediction.shape, dtype=torch.uint8)
    return_value[prediction > 0.5] = 1

    return return_value


def main_train_val():
    if (not os.path.exists(SAVE_DIR)):
        os.makedirs(SAVE_DIR)

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

    main_train_val()
