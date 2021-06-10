"""
expected command:
python experiments/finetune_resnet.py configs/finetune_config.yaml
"""
import gc
import os.path
from itertools import product
import sys

import pandas as pd
import yaml
from matplotlib import pyplot as plt

from experiments.finetune_resnet import FinetuneResnet

datasets = ["action", "car", "birds", "dtd"]
optim_modes = ["fe", "fo"]

for ds, om in product(datasets, optim_modes):

    config_file = open(os.path.join("configs", f"finetune_config.yaml"), 'r')
    config = yaml.safe_load(config_file)
    config_file.close()

    cur_config = config.copy()

    cur_config["dataset"] = f"{ds}"
    cur_config["optim_mode"] = f"{om}"

    cur_config["experiment_name"] = f"{om}_{ds}"

    print(cur_config)

    FinetuneResnet(cur_config).run()

    gc.collect()



# PLOT

# base_dir = os.path.join("/", "storage", "ssd_storage0", "results")
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
