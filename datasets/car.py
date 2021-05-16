import os
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from datasets import print_dataset_mean_std



def get_cars(*, resize=None, **kwargs):

    mean = [0.4707, 0.4601, 0.4550]
    std = [0.2667, 0.2658, 0.2706]

    dir_path = os.path.join("/", "storage", "ssd_storage0", "data", "car")

    basic_transforms = [ToTensor(), Normalize(mean, std)]

    if resize:
        basic_transforms.insert(0, Resize((resize, resize)))
    train_transforms = basic_transforms

    train = Car(dir_path, Compose(train_transforms), split="train")
    test = Car(dir_path, Compose(basic_transforms), split="test")
    return train, test


class Car(Dataset):

    def __init__(self, data_dir, transforms, split):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        data = loadmat(os.path.join(data_dir, 'cars_annos.mat'))
        if split == "train":
            test = 0
        else:
            test = 1
        self.paths = [x[0][0] for x in data['annotations'][0] if x[-1] == test]
        self.labels = [x[1][0][0] for x in data['annotations'][0] if x[-1] == test]

    def __getitem__(self, idx):
        label = self.labels[idx]
        path = self.paths[idx]
        img = Image.open(os.path.join(self.data_dir, path)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    ds = get_cars()[0]
    print_dataset_mean_std(ds)
