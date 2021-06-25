import numpy as np
import os

import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from datasets.utils import print_dataset_mean_std


def get_dtd(*, resize=None, **kwargs):
    mean = [0.5273, 0.4702, 0.4253]
    std = [0.1804, 0.1814, 0.1779]

    # dir_path = os.path.join("data", "dtd")
    dir_path = os.path.join("/", "storage", "ssd_storage0", "data", "dtd")

    basic_transforms = [ToTensor(), Normalize(mean, std)]

    if resize:
        basic_transforms.insert(0, Resize((resize, resize)))
    train_transforms = basic_transforms

    train = DTD(dir_path, Compose(train_transforms), split="train")
    test = DTD(dir_path, Compose(basic_transforms), split="test")
    return train, test


class DTD(Dataset):

    def __init__(self, data_dir, transforms, split):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        data = loadmat(os.path.join(data_dir, 'imdb', 'imdb.mat'))
        if split == "train":
            split_id = [1, 2]
        elif split == "test":
            split_id = [3]
        else:
            raise ValueError("Invalid split value")
        split = data['images'][0][0][2][0]
        self.paths = np.concatenate(data['images'][0][0][1][0][np.isin(split, split_id)])
        self.labels = data['images'][0][0][3][0][np.isin(split, split_id)] - 1
        self.data = [self.__loadimg__(i) for i in range(len(self.labels))]

    def __loadimg__(self, idx):
        path = self.paths[idx]
        img = Image.open(os.path.join(self.data_dir, 'images', path)).convert("RGB")
        return img

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).type(torch.LongTensor)
        img = self.data[idx]
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    ds = get_dtd()[0]
    print_dataset_mean_std(ds)
