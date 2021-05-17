from torchvision.models.resnet import resnet50
from datasets.cifar100 import get_cifar100
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet50_delta

class ChannelPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        pooled =  F.max_pool1d(input, self.kernel_size, self.stride,
                        self.padding, self.dilation, self.ceil_mode,
                        self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,c,w,h)


class Snow(nn.Module):
    def __init__(self, scale, out_size):
        super(Snow, self).__init__()
        self.source_activations = {}
        self.source_model = resnet50()
        self.add_hooks()
        self.freeze_model()
        self.delta_model = resnet50_delta(scale)
        self.ch_pool = ChannelPool()
    
    def add_hooks(self):
        for name, module in self.source_model.named_modules():
            def get_activation(name):
                def hook(model, ins, outs):
                    self.source_activations[name] = outs.detach().cpu().numpy()
                return hook
            module.register_forward_hook(get_activation(name))
    
    def freeze_model(self):
        for param in self.source_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        self.source_model(x)
        for k, v in self.source_activations.items():
            print(k, v.shape, self.ch_pool(v))
        # x = self.delta_model(x, self.source_activations)
        self.source_activations.clear()
        return x

    


snow_model = Snow(8, 12)

train, valid = get_cifar100()
for b in enumerate(train):
    idx, (x, y) = b
    if idx == 1:
        break
    outputs = snow_model(x)

