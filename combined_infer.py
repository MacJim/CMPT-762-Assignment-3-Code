"""
Combines detection and segmentation.
"""

import typing

import torch
import pandas as pd
import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
from detectron2.engine import DefaultPredictor

import constant.detectron
from dataset import register_datasets
from object_detection_config import get_naive_config
from object_detection_inference import infer_image


# MARK: - Combined inference
cfg = get_naive_config(train=False)
predictor = DefaultPredictor(cfg)


def get_prediction_mask(info_dict):
    filename = info_dict[constant.detectron.FILENAME_KEY]

    # Object detection
    cv_image = cv2.imread(filename)
    pred_boxes, pred_scores = infer_image(predictor, cv_image)


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
        img, true_mask, pred_mask = get_prediction_mask(sample)
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
        img, true_mask, pred_mask = get_prediction_mask(sample)
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
