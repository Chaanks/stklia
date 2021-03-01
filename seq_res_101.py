"""A ResNet implementation but using :class:`nn.Sequential`. :func:`resnet101`
returns a :class:`nn.Sequential` instead of ``ResNet``.
This code is transformed :mod:`torchvision.models.resnet`.
This code is adapted from https://gist.github.com/sublee/55ed4181e20dd59690188b279465d705
"""

from collections import OrderedDict
from typing import Iterator, Tuple, Any, List, Dict, Optional, Union

from torch import Tensor
import torch.nn as nn


Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

def resnet101(**kwargs: Any) -> nn.Sequential:
    """Constructs a ResNet-101 model."""
    return build_resnet([3, 4, 23, 3], **kwargs)

class Flatten(nn.Module):
    """Flattens any input tensor into an 1-d tensor."""

    def forward(self, x: Tensor):  # type: ignore
        return x.view(x.size(0), -1)


def build_resnet(layers: List[int],
                 emb_size: int = 256,
                 ) -> nn.Sequential:
    """Builds a ResNet as a simple sequential model.
    Note:
        The implementation is copied from :mod:`torchvision.models.resnet`.
    """
    inplanes = 64

    def make_layer(planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        nonlocal inplanes

        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(bottleneck(inplanes, planes, stride, downsample))
        inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(bottleneck(inplanes, planes))

        return nn.Sequential(*layers)

    # Build ResNet as a sequential model.
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU()),
        #('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ('layer1', make_layer(64, layers[0])),
        ('layer2', make_layer(128, layers[1], stride=2)),
        ('layer3', make_layer(256, layers[2], stride=2)),
        ('layer4', make_layer(512, layers[3], stride=2)),

        ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ('flat', Flatten()),
        ('fc', nn.Linear(512 * 4, emb_size)),
    ]))

    # Flatten nested sequentials.
    model = flatten(model)

    # Initialize weights for Conv2d and BatchNorm2d layers.
    def init_weight(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            assert isinstance(m.kernel_size, tuple)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

            m.weight.requires_grad = False
            m.weight.normal_(0, 2. / n**0.5)
            m.weight.requires_grad = True

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.weight.fill_(1)
            m.weight.requires_grad = True

            m.bias.requires_grad = False
            m.bias.zero_()
            m.bias.requires_grad = True

    model.apply(init_weight)

    return model

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Twin(nn.Module):
    def forward(self,  # type: ignore
                tensor: Tensor,
                ) -> Tuple[Tensor, Tensor]:
        return tensor, tensor


class Gutter(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self,  # type: ignore
                input_and_skip: Tuple[Tensor, Tensor],
                ) -> Tuple[Tensor, Tensor]:
        input, skip = input_and_skip
        output = self.module(input)
        return output, skip


class Residual(nn.Module):
    def __init__(self, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.downsample = downsample

    def forward(self,  # type: ignore
                input_and_identity: Tuple[Tensor, Tensor],
                ) -> Tensor:
        input, identity = input_and_identity
        if self.downsample is not None:
            identity = self.downsample(identity)
        return input + identity


def bottleneck(inplanes: int,
               planes: int,
               stride: int = 1,
               downsample: Optional[nn.Module] = None,
               ) -> nn.Sequential:
    """Creates a bottlenect block in ResNet as a :class:`nn.Sequential`."""
    layers: Dict[str, nn.Module] = OrderedDict()
    layers['twin'] = Twin()

    layers['conv1'] = Gutter(conv1x1(inplanes, planes))
    layers['bn1'] = Gutter(nn.BatchNorm2d(planes))
    layers['conv2'] = Gutter(conv3x3(planes, planes, stride))
    layers['bn2'] = Gutter(nn.BatchNorm2d(planes))
    layers['conv3'] = Gutter(conv1x1(planes, planes * 4))
    layers['bn3'] = Gutter(nn.BatchNorm2d(planes * 4))
    layers['residual'] = Residual(downsample)
    layers['relu'] = nn.ReLU()

    return nn.Sequential(layers)

def flatten(module: nn.Sequential) -> nn.Sequential:
    """Flattens a nested sequential module."""
    if not isinstance(module, nn.Sequential):
        raise TypeError('not sequential')

    return nn.Sequential(OrderedDict(_flatten(module)))

def _flatten(module: nn.Sequential) -> Iterator[Tuple[str, nn.Module]]:
    for name, child in module.named_children():
        # Flatten child sequential layers only.
        if isinstance(child, nn.Sequential):
            for sub_name, sub_child in _flatten(child):
                yield ('%s_%s' % (name, sub_name), sub_child)
        else:
            yield (name, child)
