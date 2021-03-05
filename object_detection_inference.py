"""
Part 1 inference code.
"""

import os
import random

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2

from object_detection_train import get_baseline_config
import dataset
import constant.detectron
from helper.visualization import save_visualization


def main():
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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    main()
