import math
import torch
import torch.nn as nn

from typing import Tuple, List
from torch.nn.modules.batchnorm import _BatchNorm


class BasicBlock(nn.Module):
    """The only block type DLA-34 uses (Bottleneck is only needed for DLA-46/60/102)."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        return self.relu(out)


class Root(nn.Module):
    """Merges multiple feature branches (1x1 conv over concatenated inputs)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, *x):
        x = self.conv(torch.cat(x, dim=1))
        x = self.bn(x)
        return self.relu(x)


class Tree(nn.Module):
    def __init__(self,
                 levels: int,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 level_root: bool = False,
                 root_dim: int = 0):
        super().__init__()

        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim = root_dim + in_channels

        if levels == 1:
            self.tree1 = BasicBlock(in_channels, out_channels, stride)
            self.tree2 = BasicBlock(out_channels, out_channels, 1)
            self.root = Root(root_dim, out_channels)
        else:
            self.tree1 = Tree(levels - 1, in_channels, out_channels, stride, root_dim=0)
            self.tree2 = Tree(levels - 1, out_channels, out_channels,
                              root_dim=root_dim + out_channels)

        self.level_root = level_root
        self.levels = levels

        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else None
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None, children: List[torch.Tensor] = None):
        # 'residual' is accepted only to keep Tree/BasicBlock call signatures
        # interchangeable — the actual residual used is always recomputed below.
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom

        if self.level_root:
            children.append(bottom)

        x1 = self.tree1(x, residual)

        if self.levels == 1:
            x2 = self.tree2(x1)
            return self.root(x2, x1, *children)
        else:
            children.append(x1)
            return self.tree2(x1, children=children)

class DLA34(nn.Module):
    """
    DLA-34 backbone — hardcoded, this is the only variant this project uses.
    Channels: (16, 32, 64, 128, 256, 512), Levels: (1, 1, 1, 2, 2, 1), BasicBlock only.
    """

    CHANNELS = (16, 32, 64, 128, 256, 512)
    LEVELS = (1, 1, 1, 2, 2, 1)

    def __init__(self, in_channels: int = 3):
        super().__init__()

        c = self.CHANNELS
        l = self.LEVELS

        self.base_layer = nn.Sequential(
            nn.Conv2d(in_channels, c[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=False))

        self.level0 = self._make_conv_level(c[0], c[0], l[0])
        self.level1 = self._make_conv_level(c[0], c[1], l[1], stride=2)
        self.level2 = Tree(l[2], c[1], c[2], stride=2, level_root=False)
        self.level3 = Tree(l[3], c[2], c[3], stride=2, level_root=True)
        self.level4 = Tree(l[4], c[3], c[4], stride=2, level_root=True)
        self.level5 = Tree(l[5], c[4], c[5], stride=2, level_root=True)

        self._init_weights()

    def _make_conv_level(self, inplanes: int, planes: int, num_levels: int, stride: int = 1) -> nn.Sequential:
        layers = []
        for i in range(num_levels):
            layers += [
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1,
                          padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=False)]
            inplanes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.base_layer(x)
        feats = []
        for i in range(6):
            x = getattr(self, f'level{i}')(x)
            feats.append(x)
        return tuple(feats)

    def get_out_channels(self, start_level: int) -> List[int]:
        return list(self.CHANNELS[start_level:])