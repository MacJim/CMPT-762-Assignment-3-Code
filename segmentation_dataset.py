"""
# We have provided a template data loader for your segmentation training
# You need to complete the __getitem__() function before running the code
# You may also need to add data augmentation or normalization in here
"""

import typing
import random

import torch
import torch.utils.data as data
# from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import tqdm

import constant.detectron


# MARK: - Helpers
def get_crop_coordinates(crop_x0: int, crop_y0: int, crop_width: int, crop_height: int, padding_percentage=0.1) -> typing.Tuple[int, int, int, int]:
    """
    Note that this function takes in xywh and returns xyxy.

    :param crop_x0:
    :param crop_y0:
    :param crop_width:
    :param crop_height:
    :param padding_percentage: Padding percentage on each side.
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


def get_segmentation_mask(size: typing.Tuple[int, int], segmentation_paths: typing.List[typing.List[int]]) -> Image.Image:
    """
    Get segmentation mask.

    :param size: Size of the mask in PIL format: (width, height)
    :param segmentation_paths: Segmentation paths: A list of [(x, y), (x, y), ...] or [x, y, x, y, ...]
    :return: Segmentation mask: black background, white foreground.
    """
    mask: Image.Image = Image.new("1", size, 0)    # All black by default.
    draw: ImageDraw.Draw = ImageDraw.Draw(mask)
    for segmentation_path in segmentation_paths:
        draw.polygon(segmentation_path, fill=1)

    return mask


# MARK: - Data augmentation/transform
TRANSFORM_MEAN: typing.Final = [0.485, 0.456, 0.406]
TRANSFORM_STD: typing.Final = [0.229, 0.224, 0.225]


def augment_training_image_and_mask(image: Image.Image, mask: Image.Image, size: typing.Tuple[int, int]) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    # Sometimes `image` is grayscale (mode "L").
    # We'll need to convert it to RGB.
    if (image.mode != "RGB"):
        image = image.convert("RGB")

    # Reshape.
    image = image.resize(size)    # Default is `Image.BICUBIC`
    mask = mask.resize(size, resample=Image.NEAREST)    # Mask: nearest

    # To tensor.
    image_tensor = TF.to_tensor(image)
    mask_tensor = TF.to_tensor(mask)

    # Horizontal flip.
    if (random.uniform(0.0, 1.0) < 0.5):
        image_tensor = TF.hflip(image_tensor)
        mask_tensor = TF.hflip(mask_tensor)

    # Vertical flip.
    if (random.uniform(0.0, 1.0) < 0.5):
        image_tensor = TF.vflip(image_tensor)
        mask_tensor = TF.vflip(mask_tensor)

    # Rotation.
    rotation = random.choice([0, 90, 180, 270])
    if (rotation != 0):
        image_tensor = TF.rotate(image_tensor, rotation)
        mask_tensor = TF.rotate(mask_tensor, rotation)

    # Normalize image.
    image_tensor = TF.normalize(image_tensor, TRANSFORM_MEAN, TRANSFORM_STD)

    # Remove the channel dimension of the mask.
    mask_tensor = torch.squeeze(mask_tensor, 0)

    return (image_tensor, mask_tensor)


# MARK: - Dataset
class PlaneDataset(data.Dataset):
    def __init__(self, set_name: typing.Literal["train", "val", "test"], data_dict_list: typing.List[typing.Dict], patch_width: int, patch_height: int):
        self.set_name = set_name

        self.images: typing.Dict[str, Image.Image] = {info_dict[constant.detectron.FILENAME_KEY]: Image.open(info_dict[constant.detectron.FILENAME_KEY]) for info_dict in data_dict_list}    # `open` doesn't load the images into memory.
        """Cached PIL images."""
        # Load all images into memory to prevent data loader worker contention of automatically loading images.
        # Requires ~9GB of memory.
        for image in tqdm.tqdm(self.images.values(), desc="Loaded training images", unit="images"):
            image.load()

        self.filenames_and_annotations: typing.List[typing.Tuple[str, typing.List[int], typing.List[typing.List[int]]]] = []
        """(filename, bounding box, segmentation paths)"""
        for info_dict in data_dict_list:
            filename = info_dict[constant.detectron.FILENAME_KEY]
            annotations = info_dict[constant.detectron.ANNOTATIONS_KEY]
            for annotation in annotations:
                b_box = annotation[constant.detectron.B_BOX_KEY]
                segmentation_paths = annotation[constant.detectron.SEGMENTATION_PATH_KEY]

                self.filenames_and_annotations.append((filename, b_box, segmentation_paths))

        self.patch_width = patch_width
        self.patch_height = patch_height

    def __del__(self):
        for _, image in self.images.items():
            image.close()

    # You can change the value of length to a small number like 10 for debugging of your training procedure and over-fitting make sure to use the correct length for the final training.

    def __len__(self) -> int:
        return len(self.filenames_and_annotations)

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
        # if torch.is_tensor(idx):    # Don't know what this is for.
        #     idx = idx.tolist()

        filename, b_box, segmentation_path = self.filenames_and_annotations[idx]

        full_image = self.images[filename]
        full_segmentation_mask = get_segmentation_mask(full_image.size, segmentation_path)

        crop_x0, crop_y0, crop_x1, crop_y1 = get_crop_coordinates(b_box[0], b_box[1], b_box[2], b_box[3])
        image: Image.Image = full_image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        mask: Image.Image = full_segmentation_mask.crop((crop_x0, crop_y0, crop_x1, crop_y1))

        image_tensor, mask_tensor = augment_training_image_and_mask(image, mask, (self.patch_width, self.patch_height))
        return (image_tensor, mask_tensor)


def get_plane_dataset(set_name='train', batch_size=2):
    my_data_list = DatasetCatalog.get("data_detection_{}".format(set_name))
    dataset = PlaneDataset(set_name, my_data_list)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        pin_memory=True, shuffle=True)
    return loader, dataset
