"""
expected command:
python experiments/finetune_resnet.py configs/finetune_config.yaml
"""
import os.path
from itertools import product
import sys

import pandas as pd
import yaml
from matplotlib import pyplot as plt

from experiments.finetune_resnet import FinetuneResnet

# datasets = ["action", "car", "cub_birds", "dtd"]
#
# for ds in datasets:
#     config_file = open(os.path.join("configs", f"{ds}.yaml"), 'r')
#     config = yaml.safe_load(config_file)
#     config_file.close()
#
#     # hyperparams to search
#     learning_rate = [0.005, 0.01]
#     momentum = [0, 0.9]
#     weight_decay = [0, 0.0001]
#
#     for lr, m, wd in product(learning_rate, momentum, weight_decay):
#
#         cur_config = config.copy()
#
#         experiment_name = "finetune_" + ds +\
#                           "_lr=" + str(lr) +\
#                           "_m=" + str(m) +\
#                           "_wd=" + str(wd)
#
#         cur_config["experiment_name"] = experiment_name.replace('.', '_')
#         cur_config["dataset"] = ds
#         cur_config["learning_rate"] = lr
#         cur_config["momentum"] = m
#         cur_config["weight_decay"] = wd
#
#         FinetuneResnet(cur_config).run()


# RUN GRID SEARCH

# datasets = ["action", "car", "dtd"]
# for ds in datasets:
#     config_file = open(os.path.join("configs", f"{ds}.yaml"), 'r')
#     config = yaml.safe_load(config_file)
#     config_file.close()
#
#     # hyperparams to search
#     learning_rate = [0.05, 0.1]
#     momentum = [0.9]
#     weight_decay = [0.001]
#
#     for lr, m, wd in product(learning_rate, momentum, weight_decay):
#         cur_config = config.copy()
#
#         experiment_name = cur_config["experiment_name"] + \
#                           f"_lr={str(lr)}_m={str(m)}_wd={str(wd)}"
#
#         cur_config["experiment_name"] = experiment_name.replace('.', '_')
#         cur_config["learning_rate"] = lr
#         cur_config["momentum"] = m
#         cur_config["weight_decay"] = wd
#
#         FinetuneResnet(cur_config).run()

# PLOT

# base_dir = os.path.join("/", "storage", "ssd_storage0", "results")
# learning_rate = [0.05, 0.1]
# momentum = [0.9]
# weight_decay = [0.001]
# # learning_rate = [0.05, 0.1]
# # momentum = [0.9]
# # weight_decay = [0.0001, 0.001]
# datasets = ["action", "cars", "dtd"]
# for ds in datasets:
#     print(f"plotting {ds}")
#     rows = 1
#     cols = 4
#     fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), sharey=True)
#
#     for (wd, lr, m), ax in zip(product(weight_decay, learning_rate, momentum), axs.flatten()):
#         hparams_str = f"lr={str(lr)}_m={str(m)}_wd={str(wd)}"
#         file_name = (f"finetune_{ds}_" + hparams_str).replace(".", "_")
#
#         points = pd.read_csv(os.path.join(base_dir, file_name, "metrics.csv"))
#
#         ax.plot(range(20), points['train_acc'][::2], label="train")
#         ax.plot(range(20), points['val_acc'][1::2], label="val")
#         ax.set_title(hparams_str)
#         ax.legend()
#         ax.grid()
#
#     fig.tight_layout()
#     plt.savefig(os.path.join(base_dir, f"finetune_{ds}2.png"))

# RUN TRAINING

datasets = ["action", "car", "dtd"]
for ds in datasets:
    config_file = open(os.path.join("configs", f"{ds}.yaml"), 'r')
    config = yaml.safe_load(config_file)
    config_file.close()

    FinetuneResnet(config).run()
