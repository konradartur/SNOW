import torchvision
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import pytorch_lightning.metrics.functional as FM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_resnet(**kwargs):
    return ResNet(**kwargs)


class ResNet(pl.LightningModule):

    def __init__(self, **kwargs):
        super(ResNet, self).__init__()
        self.config = kwargs
        self.size = kwargs.get('model_size', 50)
        self.pretrained = kwargs.get('pretrained', True)
        self.optim_mode = kwargs.get('optim_mode', 'ft')
        self.network = self.get_()

    def get_(self):
        if self.size is 34:
            model = torchvision.models.resnet34(pretrained=self.pretrained)
        elif self.size is 50:
            model = torchvision.models.resnet50(pretrained=self.pretrained)
        else:
            raise KeyError("wrong size defined")
        return model

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.network(x)
        loss = F.cross_entropy(pred, y)
        acc = FM.accuracy(pred, y)

        self.log_dict({
                'train_loss': loss,
                'train_acc': acc
        }, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.network(x)
        loss = F.cross_entropy(pred, y)
        acc = FM.accuracy(pred, y)

        self.log_dict({
                'val_loss': loss,
                'val_acc': acc
        }, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        if self.optim_mode == "ft":  # finetuning
            pass

        elif self.optim_mode == "fo":  # final output only aka classifier
            # freeze all parameters
            for m in self.modules():
                m.requires_grad_(False)
            # reinitialize classifier
            self.network.fc.reset_parameters()
            self.network.fc.requires_grad_(True)

        elif self.optim_mode == "fe":  # feature extractor
            filtered_keys = [x for x in self.state_dict().keys() if ('running_mean' not in x and
                                                                     'running_var' not in x and
                                                                     'num_batches_tracked' not in x)]
            for name, params in zip(filtered_keys, self.parameters()):
                if 'layer4' not in name:
                    params.requires_grad = False

            self.network.fc.reset_parameters()
            self.network.fc.requires_grad_(True)

        else:
            raise NotImplementedError("available modes: ft, fo, fe")

        optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config["learning_rate"],
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"]
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        return [optimizer], [scheduler]
