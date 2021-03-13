"""
Part 1 inference code.

Detectron2 model input/output documentation: https://detectron2.readthedocs.io/en/latest/tutorials/models.html
"""

import os
import random

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2
import numpy as np
import torch

from object_detection_config import get_baseline_config, get_naive_config
import dataset
import constant.detectron
from helper.visualization import save_visualization
from helper.patch import get_crop_patch_axes
from helper.bounding_box import nms_xyxy


def main_baseline():
    dataset.register_datasets()

    cfg = get_baseline_config(train=False)
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get(constant.detectron.TEST_DATASET_NAME)
    metadata_dict = MetadataCatalog.get(constant.detectron.TEST_DATASET_NAME)

    for i, d in enumerate(random.sample(dataset_dicts, 3)):
        im = cv2.imread(d[constant.detectron.FILENAME_KEY])
        outputs = predictor(im)    # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1], metadata=metadata_dict, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_image = out.get_image()[:, :, ::-1]

        base_filename = os.path.basename(d[constant.detectron.FILENAME_KEY])
        out_filename = os.path.join("/tmp/object_detection_inference", base_filename)
        save_visualization(out_filename, out_image)


def main_naive():
    dataset.register_datasets()

    cfg = get_naive_config(train=False)
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get(constant.detectron.TEST_DATASET_NAME)
    metadata_dict = MetadataCatalog.get(constant.detectron.TEST_DATASET_NAME)

    # for i, d in enumerate(random.sample(dataset_dicts, 1)):
    for i, d in enumerate(dataset_dicts):
        im: np.ndarray = cv2.imread(d[constant.detectron.FILENAME_KEY])

        pred_boxes = []
        pred_scores = []

        height, width, _ = im.shape
        patch_coordinates = get_crop_patch_axes(width, height, 800, 800, 200)
        for x0, x1, y0, y1 in patch_coordinates:
            patch: np.ndarray = im[y0: y1, x0: x1, :]
            output = predictor(patch)
            # print(output)    # {'instances': Instances(num_instances=2, image_height=800, image_width=800, fields=[pred_boxes: Boxes(tensor([[410.2523, 573.4615, 471.7152, 646.4885], [302.6132, 605.7527, 362.5964, 678.7469]], device='cuda:0')), scores: tensor([0.9970, 0.9954], device='cuda:0'), pred_classes: tensor([0, 0], device='cuda:0')])}

            patch_boxes: torch.Tensor = output["instances"].get("pred_boxes").tensor
            patch_scores: torch.Tensor = output["instances"].get("scores")
            for i in range(patch_boxes.shape[0]):
                patch_box = patch_boxes[i].tolist()    # Format: x0, y0, x1, y1
                patch_box = [patch_box[0] + x0, patch_box[1] + y0, patch_box[2] + x0, patch_box[3] + y0]
                patch_score = patch_scores[i].item()

                pred_boxes.append(patch_box)
                pred_scores.append(patch_score)

        visualizer = Visualizer(im[:, :, ::-1], metadata=metadata_dict, scale=0.5, instance_mode=ColorMode.IMAGE_BW)

        # Apply NMS.
        pred_boxes, pred_scores = nms_xyxy(pred_boxes, pred_scores)
        for box in pred_boxes:
            out = visualizer.draw_box(box)

        out_image = out.get_image()[:, :, ::-1]
        base_filename = os.path.basename(d[constant.detectron.FILENAME_KEY])
        out_filename = os.path.join("/tmp/object_detection_inference", base_filename)
        save_visualization(out_filename, out_image)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    # main_baseline()
    main_naive()
