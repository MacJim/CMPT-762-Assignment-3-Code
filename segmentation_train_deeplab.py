"""
Train network candidate 1: DeepLab.

https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
"""

import time
import typing
import os

import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
from torch.utils import data
from torchvision import models
import tqdm

from helper.loss_function import dice_loss
from dataset import register_datasets, get_detection_data
from segmentation_dataset import PlaneDataset
import segmentation_epoch_logger


LEARNING_RATE: typing.Final = 0.006
BATCH_SIZE: typing.Final = 4
N_CLASSES: typing.Final = 1    # Binary classification: just use 1 out channel.
N_EPOCHS: typing.Final = 200
VAL_PERCENTAGE: typing.Final = 0.15
INPUT_WIDTH: typing.Final = 256
INPUT_HEIGHT: typing.Final = 256
CHECKPOINT_SAVE_DIR: typing.Final = "seg_output"
CHECKPOINT_SAVE_EPOCH_INTERVAL: typing.Final = 10
TRAIN_LOG_FILENAME: typing.Final = os.path.join(CHECKPOINT_SAVE_DIR, "loss_log.csv")


def main():
    # MARK: Verify parameters
    if (not os.path.exists(CHECKPOINT_SAVE_DIR)):
        os.makedirs(CHECKPOINT_SAVE_DIR)
        print(f"Created checkpoint save dir `{CHECKPOINT_SAVE_DIR}`.")
    elif (os.path.isdir(CHECKPOINT_SAVE_DIR)):
        print(f"Using existing checkpoint save dir `{CHECKPOINT_SAVE_DIR}`.")
        existing_filenames = os.listdir(CHECKPOINT_SAVE_DIR)
        if existing_filenames:
            print(f"Existing checkpoint files in this directory will be overwritten.")
    else:
        raise FileExistsError(f"Checkpoint save dir `{CHECKPOINT_SAVE_DIR}` is not a folder.")

    # MARK: Variables
    network = models.segmentation.deeplabv3_resnet101(num_classes=N_CLASSES)
    network = network.cuda()

    optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    loss_function = dice_loss

    register_datasets()
    train_and_val_info_dics = get_detection_data("train")
    train_and_val_dataset = PlaneDataset("train", train_and_val_info_dics, INPUT_WIDTH, INPUT_HEIGHT)
    total_len = len(train_and_val_dataset)
    val_len = int(VAL_PERCENTAGE * total_len)
    train_len = total_len - val_len
    print(f"Patches: total: {total_len}, train: {train_len}, val: {val_len}")

    train_dataset, val_dataset = data.random_split(train_and_val_dataset, [train_len, val_len])

    # This network is very slow, so we don't need that many workers.
    train_data_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
    val_data_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    # MARK: Loop
    for epoch in range(1, N_EPOCHS + 1):
        # MARK: Train
        train_start_time = time.time()

        network.train()

        train_loss = 0.
        train_count = 0
        for image, mask in tqdm.tqdm(train_data_loader, desc="Train", unit="patches"):
            image = image.cuda()    # N, C, H, W
            mask = mask.cuda()    # N, H, W

            prediction = network(image)["out"]    # N, number of classes, H, W (shape unchanged)
            prediction = torch.squeeze(prediction, 1)    # Eliminate the channel dimension since we only have 1 output channel.
            prediction = torch.sigmoid(prediction)

            loss = loss_function(prediction, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_count += image.shape[0]

        scheduler.step()

        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        # MARK: Validation
        validation_start_time = time.time()

        network.eval()

        validation_loss = 0.
        validation_count = 0

        with torch.no_grad():
            for image, mask in val_data_loader:
                image = image.cuda()  # N, C, H, W
                mask = mask.cuda()  # N, H, W

                prediction = network(image)["out"]  # N, number of classes, H, W (shape unchanged)
                prediction = torch.squeeze(prediction, 1)  # Eliminate the channel dimension since we only have 1 output channel.
                prediction = torch.sigmoid(prediction)

                loss = loss_function(prediction, mask)
                validation_loss += loss.item()
                validation_count += image.shape[0]

        validation_end_time = time.time()
        validation_time = validation_end_time - validation_start_time

        # MARK: Log
        print(f"{epoch}. train loss: {train_loss / train_count}, validation loss: {validation_loss / validation_count}, train time: {train_time}s, validation time: {validation_time}s")
        segmentation_epoch_logger.log_epoch_details_to_file(epoch, train_loss, (train_loss / train_count), validation_loss, (validation_loss / validation_count), train_time + validation_time, TRAIN_LOG_FILENAME)

        # MARK: Save checkpoint
        if ((epoch % CHECKPOINT_SAVE_EPOCH_INTERVAL) == 0):
            checkpoint_filename = os.path.join(CHECKPOINT_SAVE_DIR, f"{epoch}.pth")
            torch.save(network.state_dict(), checkpoint_filename)
            print(f"Checkpoint saved as `{checkpoint_filename}`")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    main()
