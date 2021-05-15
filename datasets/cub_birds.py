"""
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
"""
import numpy as np
import os
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomResizedCrop
from torch.utils.data import Dataset
from PIL import Image

from datasets.utils import print_dataset_mean_std


def get_cub_birds(*, augment=True, resize=None, **kwargs):
    mean = [0.48592031, 0.49923815, 0.43139376]
    std = [0.05136569, 0.04910943, 0.06800005]

    dir_path = os.path.join("data", "birds")

    basic_transforms = [ToTensor(), Normalize(mean, std)]
    # if we augment and resize we would like to put in train_transforms RandomResizedCrop, not just Resize
    if augment:
        train_transforms = basic_transforms + [RandomHorizontalFlip()]
        val_transforms = basic_transforms
        if resize:
            train_transforms.insert(0, RandomResizedCrop((resize, resize)))
            val_transforms.insert(0, Resize((resize, resize)))
    else:
        if resize:
            basic_transforms.insert(0, Resize((resize, resize)))
        train_transforms = basic_transforms
        val_transforms = basic_transforms

    train = CubBirds(dir_path, Compose(train_transforms), split="train")
    # val = CubBirds(dir_path, Compose(val_transforms), split="val")
    test = CubBirds(dir_path, Compose(basic_transforms), split="test")
    return train, test


class CubBirds(Dataset):

    def __init__(self, data_dir, transforms, split):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms

        is_train = np.loadtxt(os.path.join(data_dir, "train_test_split.txt"), dtype=int)[:, 1]
        if split is "train":
            idxs = is_train == 1
            self.labels = np.loadtxt(os.path.join(data_dir, "image_class_labels.txt"), dtype=int)[idxs, 1]
            self.paths = np.loadtxt(os.path.join(data_dir, "images.txt"), dtype=str)[idxs, 1]
        elif split is "val":
            idxs = is_train == 0
            self.labels = np.loadtxt(os.path.join(data_dir, "image_class_labels.txt"), dtype=int)[idxs, 1][0::2]
            self.paths = np.loadtxt(os.path.join(data_dir, "images.txt"), dtype=str)[idxs, 1][0::2]
        elif split is "test":
            idxs = is_train == 0
            self.labels = np.loadtxt(os.path.join(data_dir, "image_class_labels.txt"), dtype=int)[idxs, 1][1::2]
            self.paths = np.loadtxt(os.path.join(data_dir, "images.txt"), dtype=str)[idxs, 1][1::2]
        else:
            raise RuntimeError("wrong split")

    def __getitem__(self, idx):
        label = self.labels[idx]
        path = self.paths[idx]
        img = Image.open(os.path.join(self.data_dir, "images", path)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    ds = get_cub_birds()[0]
    print_dataset_mean_std(ds)
