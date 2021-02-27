'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.pool1(self.conv1(x))))
        return x


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*out_planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*out_planes,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*out_planes)
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        # out = F.relu(out)
        return out

class ResNetNew(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetNew, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=1)
        self.fc = nn.Conv2d(
            in_channels=512*block.expansion,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [
            ConvBlock(
                in_channels=self.in_planes,
                out_channels=planes
            )
        ]
        if num_blocks == 0:
            self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            layers.append(block(planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = self.fc(out)
        out = out.view(-1, self.num_classes)
        out = F.log_softmax(out, dim=-1)
        return out


def myResNetNew():
    return ResNetNew(ResBlock, [1, 0, 1])
