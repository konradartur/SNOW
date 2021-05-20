from .action import *
from .car import *
from .birds import *
from .dtd import *


def get_dataset(name, **kwargs):
    if name == "action":
        train, test = get_action(**kwargs)
    elif name == "car":
        train, test = get_cars(**kwargs)
    elif name == "birds":
        train, test = get_birds(**kwargs)
    elif name == "dtd":
        train, test = get_dtd(**kwargs)
    else:
        raise NameError(f"dataset {name} not supported")
    return train, test
