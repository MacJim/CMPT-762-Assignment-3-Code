"""
Combines detection and segmentation.
"""

import os
import typing
import csv

import torch
import pandas as pd
import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image

import constant.detectron
from dataset import register_datasets
from segmentation_dataset import get_crop_coordinates
from segmentation_infer_unet import infer_test_patch
from helper.bounding_box import combine_bounding_boxes_naively_xyxy


# MARK: - Combined inference
OBJECT_PREDICTION_BOUNDING_BOXES_DIR: typing.Final = "/scratch/bounding_boxes"


def calculate_prediction_mask(info_dict):
    filename = info_dict[constant.detectron.FILENAME_KEY]

    # Object detection: load saved bounding box information.
    base_filename = os.path.basename(filename)
    base_filename_without_extension = os.path.splitext(base_filename)[0]
    csv_filename = os.path.join(OBJECT_PREDICTION_BOUNDING_BOXES_DIR, base_filename_without_extension + ".csv")
    with open(csv_filename, "r") as f:
        r = csv.reader(f)
        b_boxes = list(r)

    b_boxes = [[float(value) for value in pred_box] for pred_box in b_boxes]
    b_boxes = [[int(round(value)) for value in pred_box] for pred_box in b_boxes]
    b_boxes = combine_bounding_boxes_naively_xyxy(b_boxes)

    # Create return value.
    return_value = torch.zeros((info_dict[constant.detectron.HEIGHT_KEY], info_dict[constant.detectron.WIDTH_KEY]), dtype=torch.uint8)
    # Somehow the `rle_encoding` is hard-coded to use cuda tensors, so I'll need to match.
    return_value = return_value.cuda()

    # Patch predictions.
    for i, pred_box in enumerate(b_boxes):
        crop_x0, crop_y0, crop_x1, crop_y1 = get_crop_coordinates(pred_box[0], int(pred_box[1]), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1])
        patch_prediction = infer_test_patch(filename, crop_x0, crop_y0, crop_x1, crop_y1)
        patch_prediction = TF.resize(patch_prediction, (crop_y1 - crop_y0, crop_x1 - crop_x0), Image.NEAREST)    # Resize the prediction back to the original patch size.
        patch_prediction = torch.squeeze(patch_prediction, 0)    # Remove the batch dimension.
        # Remove out-of-boundary parts.
        if (crop_x0 < 0):
            delta = 0 - crop_x0
            patch_prediction = patch_prediction[:, delta:]
            crop_x0 = 0
        if (crop_x1 > info_dict[constant.detectron.WIDTH_KEY]):
            delta = crop_x1 - info_dict[constant.detectron.WIDTH_KEY]
            patch_prediction = patch_prediction[:, :-delta]
            crop_x1 = info_dict[constant.detectron.WIDTH_KEY]
        if (crop_y0 < 0):
            delta = 0 - crop_y0
            patch_prediction = patch_prediction[delta:, :]
            crop_y0 = 0
        if (crop_y1 > info_dict[constant.detectron.HEIGHT_KEY]):
            delta = crop_y1 - info_dict[constant.detectron.HEIGHT_KEY]
            patch_prediction = patch_prediction[:-delta, :]
            crop_y1 = info_dict[constant.detectron.HEIGHT_KEY]

        return_value[crop_y0: crop_y1, crop_x0: crop_x1][patch_prediction == 1] = 1

    # https://github.com/pytorch/vision/issues/1847
    # Must convert to float tensor here.
    return_value_visualization = return_value.float()
    save_image(return_value_visualization, f"/tmp/predicted_masks/{base_filename}")

    # return return_value


def get_prediction_mask_from_cache(info_dict):
    filename = info_dict[constant.detectron.FILENAME_KEY]
    base_filename = os.path.basename(filename)
    base_filename_without_extension = os.path.splitext(base_filename)[0]

    # Read cached bounding boxes.
    csv_filename = os.path.join(OBJECT_PREDICTION_BOUNDING_BOXES_DIR, base_filename_without_extension + ".csv")
    with open(csv_filename, "r") as f:
        r = csv.reader(f)
        b_boxes = list(r)

    b_boxes = [[float(value) for value in pred_box] for pred_box in b_boxes]
    b_boxes = [[int(round(value)) for value in pred_box] for pred_box in b_boxes]
    b_boxes = combine_bounding_boxes_naively_xyxy(b_boxes)

    # Read cached mask.
    cached_mask_filename = f"/tmp/predicted_masks/{base_filename}"
    cached_mask = Image.open(cached_mask_filename).convert("1")
    width, height = cached_mask.size
    cached_tensor = TF.to_tensor(cached_mask)
    cached_tensor = torch.squeeze(cached_tensor, 0)

    # Separate the planes.
    # return_value = torch.zeros(cached_tensor.shape, dtype=torch.int)
    return_value = cached_tensor.cuda()
    for i, pred_box in enumerate(b_boxes):
        instance_label = i + 1
        crop_x0, crop_y0, crop_x1, crop_y1 = get_crop_coordinates(pred_box[0], int(pred_box[1]), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1])

        crop_x0 = max(crop_x0, 0)
        crop_y0 = max(crop_y0, 0)
        crop_x1 = min(crop_x1, width)
        crop_y1 = min(crop_y1, height)

        return_value[crop_y0: crop_y1, crop_x0: crop_x1][return_value[crop_y0: crop_y1, crop_x0: crop_x1] == 1] = instance_label

    return return_value


# MARK: - Notebook: Save to CSV
# ref: https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
# https://www.kaggle.com/c/airbus-ship-detection/overview/evaluation
def rle_encoding(x):
    '''
    x: pytorch tensor on gpu, 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = torch.where(torch.flatten(x.long())==1)[0]
    if(len(dots)==0):
      return []
    inds = torch.where(dots[1:]!=dots[:-1]+1)[0]+1
    inds = torch.cat((torch.tensor([0], device=torch.device('cuda'), dtype=torch.long), inds))
    tmpdots = dots[inds]
    inds = torch.cat((inds, torch.tensor([len(dots)], device=torch.device('cuda'))))
    inds = inds[1:] - inds[:-1]
    runs = torch.cat((tmpdots, inds)).reshape((2,-1))
    runs = torch.flatten(torch.transpose(runs, 0, 1)).cpu().data.numpy()
    return ' '.join([str(i) for i in runs])


CSV_SAVE_DIR: typing.Final = "/tmp"


def save_results_to_csv():
    register_datasets()

    preddic = {"ImageId": [], "EncodedPixels": []}

    # Writing the predictions of the training set
    my_data_list = DatasetCatalog.get(constant.detectron.TRAIN_DATASET_NAME)
    for i in tqdm.tqdm(range(len(my_data_list)), position=0, leave=True):
        sample = my_data_list[i]
        sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
        calculate_prediction_mask(sample)    # NOTE: Calculate once
        pred_mask = get_prediction_mask_from_cache(sample)
        inds = torch.unique(pred_mask)
        if (len(inds) == 1):
            preddic['ImageId'].append(sample['image_id'])
            preddic['EncodedPixels'].append([])
        else:
            for index in inds:
                if (index == 0):
                    continue
                tmp_mask = (pred_mask == index)
                encPix = rle_encoding(tmp_mask)
                preddic['ImageId'].append(sample['image_id'])
                preddic['EncodedPixels'].append(encPix)

    # Writing the predictions of the test set
    my_data_list = DatasetCatalog.get(constant.detectron.TEST_DATASET_NAME)
    for i in tqdm.tqdm(range(len(my_data_list)), position=0, leave=True):
        sample = my_data_list[i]
        sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
        calculate_prediction_mask(sample)    # NOTE: Calculate once
        pred_mask = get_prediction_mask_from_cache(sample)
        inds = torch.unique(pred_mask)
        if (len(inds) == 1):
            preddic['ImageId'].append(sample['image_id'])
            preddic['EncodedPixels'].append([])
        else:
            for j, index in enumerate(inds):
                if (index == 0):
                    continue
                tmp_mask = (pred_mask == index).double()
                encPix = rle_encoding(tmp_mask)
                preddic['ImageId'].append(sample['image_id'])
                preddic['EncodedPixels'].append(encPix)

    pred_file = open("{}/pred.csv".format(CSV_SAVE_DIR), 'w')
    pd.DataFrame(preddic).to_csv(pred_file, index=False)
    pred_file.close()


if __name__ == '__main__':
    save_results_to_csv()
