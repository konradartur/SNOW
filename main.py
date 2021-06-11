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


# datasets = ["action", "car", "dtd", "birds"]
datasets = ["action", "car"]
# datasets = ["dtd", "birds"]
optim_modes = ["fe"]
# optim_modes = ["ft", "fe", "fo"]


def make_config(dataset, optim_mode):
    config_file = open(os.path.join("configs", f"finetune_config.yaml"), 'r')
    config = yaml.safe_load(config_file)
    config_file.close()

    cur_config = config.copy()

    cur_config["dataset"] = f"{dataset}"
    cur_config["optim_mode"] = f"{optim_mode}"
    cur_config["experiment_name"] = f"{optim_mode}_{dataset}"

    return cur_config


def run_ft_fo_fe():
    for ds, om in product(datasets, optim_modes):

        cur_config = make_config(ds, om)

        print(cur_config)
        FinetuneResnet(cur_config).run()
        gc.collect()


def plot():
    rows = 1
    cols = len(datasets)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), sharey=True)

    for idx, (ds, ax) in enumerate(zip(datasets, axs)):

        for om in optim_modes:

            cur_config = make_config(ds, om)
            base_dir = os.path.join(os.environ["RESULTS_DIR"], cur_config["experiment_name"])
            points = pd.read_csv(os.path.join(base_dir, "metrics.csv"))
            # ax.plot(range(cur_config["n_epochs"]), points['train_acc'][::2], label="train")
            ax.plot(range(cur_config["n_epochs"]), points['val_acc'][1::2][:int(cur_config["n_epochs"])], label=f"{om}")

        # add snow
        points = pd.read_csv(os.path.join(os.environ["RESULTS_DIR"], "results.csv"))
        print(points)
        print(points.loc[[f"{ds}"]])
        ax.plot(90, points.iloc[f"{ds}"], label=f"snow")

        ax.set_title(f"{ds}")
        if idx == 0:
            ax.legend()
        ax.grid()

    fig.tight_layout()
    plt.savefig(os.path.join(os.path.join(os.environ["RESULTS_DIR"], "plot.png")))


def filters():
    cur_config = make_config("birds", "fe")
    FinetuneResnet(cur_config).visualize_filters()


if __name__ == "__main__":
    # run_ft_fo_fe()
    plot()
    # filters()
