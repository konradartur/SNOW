import os
import time

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from datasets import get_dataset
from models import plain_resnet


class FinetuneResnet:

    def __init__(self, config):
        self.config = config
        self.save_path = os.path.join("/", "storage", "ssd_storage0", "results", config["experiment_name"])

    def run(self):
        train, val = get_dataset(self.config["dataset"], **self.config)
        model = plain_resnet.get_resnet(**self.config)

        train_loader = DataLoader(
                train,
                num_workers=4,
                batch_size=self.config["batch_size"],
                shuffle=True
        )

        val_loader = DataLoader(
                val,
                num_workers=4,
                batch_size=self.config["batch_size"],
                shuffle=False
        )

        gpus = 1
        trainer = Trainer(
                default_root_dir=self.save_path,
                max_epochs=self.config["n_epochs"],
                gpus=gpus if torch.cuda.is_available() else 0,
                logger=CSVLogger(self.save_path, "", ""),
                checkpoint_callback=False
        )

        start_time = time.time()
        trainer.fit(model, train_loader, val_loader)
        end_time = time.time()

        f = open(os.path.join(self.save_path, 'training_duration.txt'), 'w')
        f.write(str(end_time - start_time))
        f.close()

    def report(self, config):
        pass
