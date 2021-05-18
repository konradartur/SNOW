from torchvision.models.resnet import resnet50
from datasets.cifar100 import get_cifar100
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet50_delta


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


class ChannelPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.params = torch.rand(in_channels, requires_grad=True)

    def forward(self, input):
        vals, indices = torch.topk(self.params + torch.rand(self.in_channels), self.out_channels)
        batch_size, num_channels, width, height = input.shape
        result = vals * input[:, indices, :, :].view(batch_size, width, height, self.out_channels)
        return result.view(batch_size, self.out_channels, width, height)
    

chp = ChannelPool(16, 2)

A = torch.rand((64, 16, 8, 8))
res = chp(A)
print(res.shape)

# snow_model = Snow(8, 12)

# train, valid = get_cifar100()
# for b in enumerate(train):
#     idx, (x, y) = b
#     if idx == 1:
#         break
#     outputs = snow_model(x)

