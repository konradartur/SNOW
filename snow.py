from torchvision.models.resnet import resnet50
from datasets.cifar100 import get_cifar100
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet50_delta


def iterdict(d):
  for k,v in d.items():        
     if isinstance(v, dict):
         iterdict(v)
     else:            
         print (k)

class Snow(nn.Module):
    def __init__(self, scale, out_size):
        super(Snow, self).__init__()
        self.source_activations = {}
        self.source_model = resnet50()
        self.add_hooks()
        self.freeze_model()
        self.delta_model = resnet50_delta(scale)
        # self.ch_pool = ChannelPool()
    
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
    
    def __ugly_ass_function(self):
        result_dict = {}
        for k, v in self.source_activations.items():
            v = 1
            res = k.split('.')
            if len(res) == 1:
                result_dict[k] = v
                continue

            layer_name = res[0] + "_inner"
            bottleneck_id = res[1]
            if layer_name not in result_dict:
                result_dict[layer_name] = {}
            
            if len(res) == 2:
                result_dict[layer_name][bottleneck_id] = v
                continue
            bottleneck_id += "_inner"
            if bottleneck_id not in result_dict[layer_name]:
                result_dict[layer_name][bottleneck_id] = {}
            module_name = res[2]
            if len(res) == 4:
                module_name = res[2] + res[3]
            result_dict[layer_name][bottleneck_id][module_name] = v
        return result_dict

    def __get_feature_maps(self, x):
        self.source_model(x)
        feature_map = self.__ugly_ass_function()
        self.source_activations.clear()
        return feature_map

    def forward(self, x):
        feature_map = self.__get_feature_maps(x)
        x = self.delta_model(x, feature_map)
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
    

# chp = ChannelPool(16, 2)

# A = torch.rand((64, 16, 8, 8))
# res = chp(A)
# concated = torch.cat((res, A), dim=1)

snow_model = Snow(8, 12)
print(snow_model)

train, valid = get_cifar100()
for b in enumerate(train):
    idx, (x, y) = b
    if idx == 1:
        break
    outputs = snow_model(x)

