import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from datasets import print_dataset_mean_std


def get_action(*, resize=None, **kwargs):
    mean = [0.4670, 0.4409, 0.4021]
    std = [0.2465, 0.2389, 0.2429]

    # dir_path = os.path.join("/", "storage", "ssd_storage0", "data", "action")
    dir_path = os.path.join(os.environ["DATA_DIR"], "action")

    basic_transforms = [ToTensor(), Normalize(mean, std)]

    if resize:
        basic_transforms.insert(0, Resize((resize, resize)))
    train_transforms = basic_transforms

    train = Action(dir_path, Compose(train_transforms), split="train")
    test = Action(dir_path, Compose(basic_transforms), split="test")
    return train, test


class Action(Dataset):

    def __init__(self, data_dir, transforms, split):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        data = []
        for file in os.listdir(os.path.join(data_dir, 'ImageSplits'))[1:]:
            if file not in ["train.txt", "test.txt"]:
                with open(os.path.join(data_dir, 'ImageSplits', file)) as path_list:
                    for path in path_list:
                        data.append([path[:-1], '_'.join(file.split('_')[:-1]), file.split('_')[-1].split('.')[0]])
        data = np.array(data)

        if split not in ['test','train']:
            raise ValueError('Invalid split argument.')

        filter = data[:, 2] == split
        self.paths = data[:, 0][filter]
        # %%
        labels_list = np.unique(data[:, 1]).tolist()
        self.labels = np.array([labels_list.index(x) + 1 for x in data[:, 1][filter]])
        self.data = [self.__loadimg__(i) for i in range(len(self.labels))]

    def __loadimg__(self, idx):
        path = self.paths[idx]
        img = Image.open(os.path.join(self.data_dir, 'JPEGImages', path)).convert("RGB")
        return img

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.data[idx]
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    ds = get_action()[0]
    print_dataset_mean_std(ds)

