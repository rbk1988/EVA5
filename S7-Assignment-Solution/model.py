"""Define the class file for the model.."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=padding)

        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class CIFARClassifier(nn.Module):
    def __init__(self):
        super(CIFARClassifier, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 2, dilation=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )# input 32x32x3 output 32x32x32 RF 5x5
        # 3x3 dilated effectively has RF of 5x5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 2, dilation=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )# input 32x32x3 output 32x32x32 RF 9x9
        self.pool1 = nn.MaxPool2d(2, 2)# input 32x32x32 output 16x16x32 RF 18x18
        self.convblock3 = nn.Sequential(
            SeparableConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )# input 16x16x32 output 16x16x64 RF 20x20
        self.convblock4 = nn.Sequential(
            SeparableConv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )# input 16x16x64 output 16x16x64 RF 22x22
        self.pool2 = nn.MaxPool2d(2, 2)# input 16x16x64 output 8x8x64 RF 44x44
        self.convblock5 = nn.Sequential(
            SeparableConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )# input 8x8x64 output 8x8x128 RF 46x46
        self.convblock6 = nn.Sequential(
            SeparableConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )# input 8x8x128 output 8x8x128 RF 48x48
        self.pool3 = nn.MaxPool2d(2, 2)# input 8x8x128 output 4x4x128 RF 96x96
        self.convblock7 = nn.Sequential(
            SeparableConv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )# input 4x4x128 output 4x4x256 RF 98x98
        self.convblock8 = nn.Sequential(
            SeparableConv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )# input 4x4x256 output 4x4x256 RF 100x100
        self.gap = nn.AvgPool2d(kernel_size=4)#input 4x4x32 output 1x1x32 RF ?
        self.conv9 = nn.Conv2d(256, 10, 1, bias=False)


    def forward(self, x):
      """."""
      x = self.convblock1(x)
      x = self.convblock2(x)
      x = self.pool1(x)

      x = self.convblock3(x)
      x = self.convblock4(x)
      x = self.pool2(x)
     
      x = self.convblock5(x)
      x = self.convblock6(x)
      x = self.pool3(x)

      x = self.convblock7(x)
      x = self.convblock8(x)      

      x = self.gap(x)

      x = self.conv9(x)
      x = x.view(-1, 10)
      x = F.log_softmax(x,dim=1)
      return x
