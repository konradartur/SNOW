from torchvision.models.resnet import resnet50
from datasets.cifar100 import get_cifar100
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet50_delta



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

    

# chp = ChannelPool(16, 2)

# A = torch.rand((64, 16, 8, 8))
# res = chp(A)
# concated = torch.cat((res, A), dim=1)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    model = Snow(8, 2, 12)
    learning_rate = 0.1
    train_dataloader, test_dataloader = get_cifar100()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
