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
    def __init__(self, K, M, out_size):
        super(Snow, self).__init__()
        self.source_activations = {}
        self.source_model = resnet50()
        self.add_hooks()
        self.freeze_model()
        self.delta_model = resnet50_delta(K, M)
        # self.ch_pool = ChannelPool()
    
    def add_hooks(self):
        for name, module in self.source_model.named_modules():
            def get_activation(name):
                def hook(model, ins, outs):
                    self.source_activations[name] = outs.detach()
                return hook
            module.register_forward_hook(get_activation(name))
    
    def freeze_model(self):
        for param in self.source_model.parameters():
            param.requires_grad = False
    
    def __ugly_ass_function(self):
        result_dict = {}
        for k, v in self.source_activations.items():
            print(k)
            # v = 1
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
        # print(result_dict)
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

    

# chp = ChannelPool(16, 2)

# A = torch.rand((64, 16, 8, 8))
# res = chp(A)
# concated = torch.cat((res, A), dim=1)

snow_model = Snow(8, 2, 12)
# print(snow_model)

train, valid = get_cifar100()
for b in enumerate(train):
    idx, (x, y) = b
    if idx == 1:
        break
    outputs = snow_model(x)

