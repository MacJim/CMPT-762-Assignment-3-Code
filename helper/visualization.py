import os

import cv2


def save_visualization(filename: str, image):
    parent_dir = os.path.dirname(filename)
    if (not os.path.exists(parent_dir)):
        os.makedirs(parent_dir)

    result = cv2.imwrite(filename, image)
    if result:
        print(f"Visualization saved to {filename}")
    else:
        print(f"Failed to save visualization to {filename}")
