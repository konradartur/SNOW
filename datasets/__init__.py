from .cifar100 import get_cifar100
from .action import *
from .car import *
from .birds import *
from .dtd import *
from .food import *


def get_dataset(name, **kwargs):
    if name == "action":
        train, test = get_action(**kwargs)
    elif name == "car":
        train, test = get_cars(**kwargs)
    elif name == "birds":
        train, test = get_birds(**kwargs)
    elif name == "dtd":
        train, test = get_dtd(**kwargs)
    elif name == "food":
        train, test = get_food(**kwargs)
    else:
        raise NameError(f"dataset {name} not supported")
    return train, test
