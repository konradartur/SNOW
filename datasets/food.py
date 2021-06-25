import os

import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from datasets.utils import print_dataset_mean_std


def get_food(*, resize=None, **kwargs):
    mean = [0.5450, 0.4435, 0.3436]
    std = [0.2335, 0.2443, 0.2424]

    dir_path = os.path.join("/", "storage", "ssd_storage0", "data", "food")
    dir_path = os.path.join(os.environ["DATA_DIR"], "food")

    basic_transforms = [ToTensor(), Normalize(mean, std)]

    if resize:
        basic_transforms.insert(0, Resize((resize, resize)))
    train_transforms = basic_transforms

    train = Food(dir_path, Compose(train_transforms), split="train")
    test = Food(dir_path, Compose(basic_transforms), split="test")
    return train, test


class Food(Dataset):

    def __init__(self, data_dir, transforms, split):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms

        if split == "train":
            with open(os.path.join(self.data_dir, "meta", "train.json")) as file:
                data = json.load(file)
        elif split == "test":
            with open(os.path.join(self.data_dir, "meta", "test.json")) as file:
                data = json.load(file)
        else:
            raise ValueError("Invalid split value")
        label_list = list(data.keys())
        self.paths = [path + ".jpg" for paths in data.values() for path in paths]
        self.labels = [label_list.index(label) for label, paths in data.items() for _ in paths]
        # self.data = [self.__loadimg__(i) for i in range(len(self.labels))]
        # Too big dataset to load whole into memory :(

    def __loadimg__(self, idx):
        path = self.paths[idx]
        img = Image.open(os.path.join(self.data_dir, 'images', path)).convert("RGB")
        return img

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).type(torch.LongTensor)
        # img = self.data[idx]
        img = self.__loadimg__(idx)
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    ds = get_food()[0]
    print_dataset_mean_std(ds)
