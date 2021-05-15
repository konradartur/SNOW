import torchvision
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import pytorch_lightning.metrics.functional as FM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_resnet():
    return ResNet


class ResNet(pl.LightningModule):

    def __init__(self, **kwargs):
        super(ResNet, self).__init__()
        self.config = kwargs
        self.size = kwargs["model_kwargs"]['size']
        self.pretrained = kwargs['model_kwargs'].get('pretrained', True)
        self.optim_name = kwargs['optim_kwargs']['name']
        self.network = self.get_()

    def get_(self):
        model = None
        if self.size is 18:
            model = torchvision.models.resnet18(pretrained=self.pretrained)
        elif self.size is 34:
            model = torchvision.models.resnet34(pretrained=self.pretrained)
        elif self.size is 50:
            model = torchvision.models.resnet50(pretrained=self.pretrained)
        elif self.size is 101:
            model = torchvision.models.resnet101(pretrained=self.pretrained)
        elif self.size is 152:
            model = torchvision.models.resnet152(pretrained=self.pretrained)
        else:
            KeyError("wrong size defined")
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
        pass
        # optim = Runtime.optimizers[self.optim_name]
        # return optim(self.parameters(), **self.config)
