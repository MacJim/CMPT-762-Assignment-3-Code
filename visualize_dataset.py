"""
Visualize 3 random samples of the training data using `detectron2.utils.visualizer`.

Refer to the official tutorial: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=UkNbUzUOLYf0&line=1&uniqifier=1
"""

import os
import random

import cv2
import detectron2.data
from detectron2.utils.visualizer import Visualizer

import dataset
import constant.detectron


def main():
    dataset.register_datasets()

    train_dataset = detectron2.data.DatasetCatalog.get(constant.detectron.TRAIN_DATASET_NAME)
    train_metadata = detectron2.data.MetadataCatalog.get(constant.detectron.TRAIN_DATASET_NAME)
    for i, info_dict in enumerate(random.sample(train_dataset, 3)):
        image = cv2.imread(info_dict[constant.detectron.FILENAME_KEY])
        visualizer = Visualizer(image[:, :, ::-1], metadata=train_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(info_dict)
        out_image = out.get_image()[:, :, ::-1]

        out_filename = f"/tmp/{i + 1}.png"
        cv2.imwrite(out_filename, out_image)
        print(f"Visualization saved to {out_filename}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    main()
