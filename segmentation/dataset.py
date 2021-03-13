"""
# We have provided a template data loader for your segmentation training
# You need to complete the __getitem__() function before running the code
# You may also need to add data augmentation or normalization in here
"""

import torch
import torch.utils.data as data


class PlaneDataset(data.Dataset):
    def __init__(self, set_name, data_list):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # Converting the image to tensor and change the image format (Channels-Last => Channels-First)
        ])
        self.set_name = set_name
        self.data = data_list
        self.instance_map = []
        for i, d in enumerate(self.data):
            for j in range(len(d['annotations'])):
                self.instance_map.append([i, j])

    '''
    # you can change the value of length to a small number like 10 for debugging of your training procedure and overfeating
    # make sure to use the correct length for the final training
    '''

    def __len__(self):
        return len(self.instance_map)

    def numpy_to_tensor(self, img, mask):
        if self.transforms is not None:
            img = self.transforms(img)
        img = torch.tensor(img, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.float)
        return img, mask

    '''
    # Complete this part by using get_instance_sample function
    # make sure to resize the img and mask to a fixed size (for example 128*128)
    # you can use "interpolate" function of pytorch or "numpy.resize"
    # TODO: 5 lines
    '''

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.instance_map[idx]
        data = self.data[idx[0]]

        return img, mask


def get_plane_dataset(set_name='train', batch_size=2):
    my_data_list = DatasetCatalog.get("data_detection_{}".format(set_name))
    dataset = PlaneDataset(set_name, my_data_list)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        pin_memory=True, shuffle=True)
    return loader, dataset
