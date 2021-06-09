# source: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet50
# with some modifications 
import torch
from torch import Tensor
import torch.nn as nn
import math
from typing import Type, Any, Callable, Union, List, Optional, Dict


device = "cuda"

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ChannelPool(nn.Module):
    def __init__(self, in_channels, out_channels, variance=0.01):
        super(ChannelPool, self).__init__()
        # print("ChPool:", in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.params = torch.randn((in_channels,), requires_grad=True).to(device)
        self.variance = variance

    def forward(self, x):
        if self.training:
            rand_v = torch.normal(0.0, math.sqrt(self.variance), (self.in_channels,)).to(device)
            # print("tutaj", rand_v.shape)
            _, indices = torch.topk(self.params + rand_v, self.out_channels)
            sel_weight = self.params[indices]
        else:
            sel_weight, indices = torch.topk(self.params, self.out_channels)
        result = sel_weight.view(1,-1,1,1) * x[:, indices, :, :]
        return result


class BasicBlock(nn.Module):
    pass


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        K: int,
        M: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        variance: float = 0.01,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        inplanes_scaled = inplanes // K
        planes_scaled = planes // K
        width_scaled = width // K
        print(variance)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # print(inplanes, width, K, M)
        self.M = M
        self.K = K
        self.conv1 = conv1x1(inplanes_scaled + inplanes // M, width_scaled)
        self.conv1_chp = ChannelPool(width, width // M, variance)
        self.bn1 = norm_layer(width_scaled + width // M)
        self.conv2 = conv3x3(width_scaled + width // M, width_scaled, stride, groups, dilation)
        self.conv2_chp = ChannelPool(width, width // M, variance)
        self.bn2 = norm_layer(width_scaled + width // M)
        self.conv3 = conv1x1(width_scaled + width // M, planes_scaled * self.expansion)
        self.conv3_chp = ChannelPool(planes * self.expansion, planes * self.expansion // M, variance)
        self.bn3 = norm_layer(planes_scaled * self.expansion + planes * self.expansion // M)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.downsample_chp = ChannelPool(planes * self.expansion , planes * self.expansion // M, variance)
        self.stride = stride

    def _apply_chp(self, chp, t, x):
        return torch.cat((chp(t), x), dim=1)

    def forward(self, x: Tensor, feature_map: Dict) -> Tensor:
        identity = x
        # print(feature_map.keys())
        # print("Before conv1", x.shape)
        out = self.conv1(x)
        t = feature_map["conv1"]
        # print("After conv1", out.shape)
        out = self._apply_chp(self.conv1_chp, t, out)

        out = self.bn1(out)
        out = self.relu(out)

        # print("Before conv2", out.shape)
        out = self.conv2(out)
        t = feature_map["conv2"]
        # print("After conv2", out.shape)
        out = self._apply_chp(self.conv2_chp, t, out)

        out = self.bn2(out)
        out = self.relu(out)

        # print("Before conv3", out.shape)
        out = self.conv3(out)
        # print("After conv3", out.shape)
        t = feature_map["conv3"]
        # print("last t shape", t.shape)
        out = self._apply_chp(self.conv3_chp, t, out)
        # print("After last channel pool", out.shape)

        out = self.bn3(out)

        # print("out before downsample", out.shape)
        if self.downsample is not None:
            identity = self.downsample(x)
            t = feature_map["downsample"]
            identity = self._apply_chp(self.downsample_chp, t, identity)
        # print("out shape", out.shape)
        # print("identity shape", identity.shape)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        K: int,
        M: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        variance = 0.01,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes // K, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_chp = ChannelPool(self.inplanes, self.inplanes // M, variance=variance)
        self.bn1 = norm_layer(self.inplanes // K + self.inplanes // M)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], K, M, variance=variance)
        self.layer2 = self._make_layer(block, 128, layers[1], K, M, stride=2,
                                       dilate=replace_stride_with_dilation[0], variance=variance)
        self.layer3 = self._make_layer(block, 256, layers[2], K, M, stride=2,
                                       dilate=replace_stride_with_dilation[1], variance=variance)
        self.layer4 = self._make_layer(block, 512, layers[3], K, M, stride=2,
                                       dilate=replace_stride_with_dilation[2], variance=variance)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.mod_layers = nn.Sequential(*[*self.layer1, *self.layer2, *self.layer3, *self.layer4])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        fc_size = 512 * block.expansion
        self.fc = nn.Linear(fc_size // K + fc_size // M , num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, K: int, M: int,
                    stride: int = 1, dilate: bool = False, variance=0.01) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes // K + self.inplanes // M, planes * block.expansion // K, stride),
                norm_layer(planes * block.expansion // K),
            )

        layers = []
        layers.append(block(self.inplanes, planes, K, M, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, variance=variance))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, K, M, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, variance=variance))

        return layers
    
    def _apply_chp(self, chp, t, x):
        return torch.cat((chp(t), x), dim=1)

    def _forward_impl(self, x: Tensor, feature_map: Dict) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        t = feature_map["conv1"]
        x = self._apply_chp(self.conv1_chp, t, x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print("Before layers x", x.shape)

        # for block in 
        for idx, layer in enumerate(self.layers):
            layer_name = "layer{}_inner".format(idx+1)
            for idy, block in enumerate(layer):
                block_name = "{}_inner".format(idy)
                x = block(x, feature_map[layer_name][block_name])

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor, feature_map: Dict) -> Tensor:
        return self._forward_impl(x, feature_map)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    K: int,
    M: int,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(K, M, block, layers, **kwargs)
    return model

def resnet50_delta(K: int, M: int, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], K, M, progress,
                   **kwargs)

