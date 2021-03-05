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

        out_filename = f"/tmp/{i + 1}.png"
        cv2.imwrite(out_filename, out.get_image()[:, :, ::-1])
        print(f"Visualization saved to {out_filename}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    main()
