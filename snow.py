from tqdm import tqdm
from torchvision.models.resnet import resnet50
from datasets.car import get_cars
from datasets.birds import get_birds
from datasets.action import get_action
from datasets.dtd import get_dtd
from datasets.food import get_food
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time
from models import resnet50_delta


class Snow(nn.Module):
    def __init__(self, K, M, out_size, variance=0.01):
        super(Snow, self).__init__()
        self.source_activations = {}
        self.source_model = resnet50(pretrained=True)
        self.add_hooks()
        self.freeze_model()
        self.delta_model = resnet50_delta(K, M, num_classes=out_size, variance=variance)

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
        self.source_activations = {}
        self.source_model(x)
        feature_map = self.__ugly_ass_function()
        return feature_map

    def forward(self, x):
        feature_map = self.__get_feature_maps(x)
        x = self.delta_model(x, feature_map)
        return x


def iterdict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            iterdict(v)
        else:
            print(k, ":", v.shape)


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X = X.to(device)
        y = y.to(device)
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


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100 * correct


if __name__ == "__main__":
    device = "cuda:1"
    dataset_names = ["dtd", "action", "birds", "cars"]  # , "food"]
    dataset_classes = [47, 40, 200, 196]  # , 101]
    dataset_load_function = [get_dtd, get_action, get_birds, get_cars]  # , get_food]
    for i in range(len(dataset_names)):
        if device != "cpu":
            torch.cuda.empty_cache()
        model = Snow(8, 8, dataset_classes[i], variance=0.001).to(device)
        print("Model created")
        train_dataset, test_dataset = dataset_load_function[i](resize=224)
        print(f"Dataset {dataset_names[i]} loaded")
        batch_size = 64
        print("Preparing dataloader")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        learning_rate = 1
        momentum = 0.9
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0001)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        epochs = 50
        test_time = []
        accuracies = []
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, device)
            scheduler.step()
            start_test = time.time()
            acc = test_loop(test_dataloader, model, loss_fn, device)
            end_test = time.time()
            if len(accuracies) == 0 or acc > max(accuracies):
                torch.save(model.state_dict(), f"SNOW_{dataset_names[i]}.pth")
            accuracies.append(acc)
            test_time.append(end_test - start_test)
        print("Done!")
        print(f"Mean evaluation time: {np.array(test_time).mean()}")
        with open(f"SNOW_{dataset_names[i]}.txt", "w") as file:
            file.write(f"{str(accuracies)}\n")
            file.write(f"{str(test_time)}\n")
        del model
