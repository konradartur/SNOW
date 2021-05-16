"""
expected command:
python experiments/finetune_resnet.py configs/finetune_config.yaml
"""
import os.path
from itertools import product
import sys
import yaml
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


datasets = ["action", "car", "birds", "dtd"]

config_file = open(os.path.join("configs", f"dtd.yaml"), 'r')
config = yaml.safe_load(config_file)
config_file.close()

FinetuneResnet(config).run()

