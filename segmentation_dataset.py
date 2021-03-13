"""
# We have provided a template data loader for your segmentation training
# You need to complete the __getitem__() function before running the code
# You may also need to add data augmentation or normalization in here
"""

import typing

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

import constant.detectron


# MARK: - Helpers
def get_crop_coordinates(crop_x0: int, crop_y0: int, crop_width: int, crop_height: int, padding_percentage=0.2) -> typing.Tuple[int, int, int, int]:
    """
    Note that this function takes in xywh and returns xyxy.

    :param crop_x0:
    :param crop_y0:
    :param crop_width:
    :param crop_height:
    :param padding_percentage:
    :return: (crop_x0, crop_y0, crop_x1, crop_y1)
    """
    crop_x1 = crop_x0 + crop_width
    crop_y1 = crop_y0 + crop_height

    # Unused: Make sure the cropped region is a square.
    # if (crop_width >= crop_height):
    #     crop_side_length = crop_width
    #     delta = crop_width - crop_height
    #     crop_y0 -= (delta // 2)
    #     crop_y1 += (delta // 2)
    # else:
    #     crop_side_length = crop_height
    #     delta = crop_height - crop_width
    #     crop_x0 -= (delta // 2)
    #     crop_x1 += (delta // 2)

    if (padding_percentage > 0):
        padding_on_each_side = int(max(crop_width, crop_height) * padding_percentage)
        crop_x0 -= padding_on_each_side
        crop_y0 -= padding_on_each_side
        crop_x1 += padding_on_each_side
        crop_y1 += padding_on_each_side

    return (crop_x0, crop_y0, crop_x1, crop_y1)


# MARK: - Transforms
TRAIN_TRANSFORMS: typing.Final = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])


# MARK: - Dataset
class PlaneDataset(data.Dataset):
    def __init__(self, set_name: typing.Literal["train", "val", "test"], data_dict_list: typing.List[typing.Dict]):
        self.set_name = set_name

        self.images: typing.Dict[str, Image.Image] = {info_dict[constant.detectron.FILENAME_KEY]: Image.open(info_dict[constant.detectron.FILENAME_KEY]) for info_dict in data_dict_list}
        """Cached PIL images."""

        self.annotations: typing.List[typing.Tuple[str, typing.List[int], typing.List[typing.List[int]]]] = []
        """(filename, bounding box, segmentation paths)"""

    # You can change the value of length to a small number like 10 for debugging of your training procedure and over-fitting make sure to use the correct length for the final training.

    def __len__(self) -> int:
        return len(self.annotations)

    # def numpy_to_tensor(self, img, mask):
    #     if self.transforms is not None:
    #         img = self.transforms(img)
    #     img = torch.tensor(img, dtype=torch.float)
    #     mask = torch.tensor(mask, dtype=torch.float)
    #     return img, mask

    # Complete this part by using get_instance_sample function
    # make sure to resize the img and mask to a fixed size (for example 128*128)
    # you can use "interpolate" function of pytorch or "numpy.resize"
    # TODO: 5 lines

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        idx = self.instance_map[idx]
        data = self.data[idx[0]]

        return img, mask


def get_plane_dataset(set_name='train', batch_size=2):
    my_data_list = DatasetCatalog.get("data_detection_{}".format(set_name))
    dataset = PlaneDataset(set_name, my_data_list)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        pin_memory=True, shuffle=True)
    return loader, dataset
