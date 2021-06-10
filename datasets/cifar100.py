import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision
import os

from os.path import join


def get_cifar100(image_size=None, batch_size=128):
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    DATA_DIR = os.environ["DATA_DIR"]

    transforms_train = []
    transforms_test = []
    if image_size:
        transforms_train.append(T.Resize(image_size))
        transforms_test.append(T.Resize(image_size))
    
    transform_train = T.Compose(transforms_train + [
        T.ToTensor(),
        T.Normalize(cifar_mean, cifar_std)
    ])
    transform_test = T.Compose(transforms_test + [
        T.ToTensor(),
        T.Normalize(cifar_mean, cifar_std)
    ])

    print(transform_train)

    trainval = torchvision.datasets.CIFAR10(
            root=join(DATA_DIR, 'cifar100'), train=True, download=True, transform=transform_train)
    test = torchvision.datasets.CIFAR10(
        root=join(DATA_DIR, 'cifar100'), train=False, download=True, transform=transform_test)
    
    train_dataloader = DataLoader(trainval, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader