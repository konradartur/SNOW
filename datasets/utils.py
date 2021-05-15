from tqdm import tqdm
import torch
from torch.utils.data import Dataset


def print_dataset_mean_std(ds: Dataset):
    std = []
    mean = []
    for i, (x, y) in enumerate(tqdm(ds)):
        x = x.view((3, -1))
        std.append(x.std(axis=1))
        mean.append(x.mean(axis=1))
    print(f"Std: {torch.stack(std).mean(axis=0)}")
    print(f"Mean: {torch.stack(mean).mean(axis=0)}")
