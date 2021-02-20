import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFARClassifier(nn.Module):
    """."""
    def __init__(self):
        super(CIFARClassifier, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU()
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU()
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU()
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU()
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU()
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU()
        )
        self.gap = nn.AvgPool2d(kernel_size=8)
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 3, stride = 1, padding = 1, bias=False),
        )

    def forward(self, x):
        """."""
        x1 = x # input:32x32x3 output: ?
        x2 = self.convblock1(x1)# input:32x32x3 output: 32x32x3
        x3 = self.convblock2(torch.sum(torch.stack([x1,x2]), dim=0))# input:32x32x3 output: 32x32x3
        
        x4 = self.pool1(torch.sum(torch.stack([x1,x2,x3]), dim=0))# input:32x32x3 output: 16x16x3

        x5 = self.convblock3(x4)# input:16x16x32 output: 16x16x32
        x6 = self.convblock4(torch.sum(torch.stack([x4,x5]), dim=0))# input:16x16x32 output: 16x16x32
        x7 = self.convblock5(torch.sum(torch.stack([x4,x5,x6]), dim=0))# input:16x16x32 output: 16x16x32
        x8 = self.pool2(torch.sum(torch.stack([x5,x6,x7]), dim=0))# input:16x16x32 output: 8x8x32

        x9 = self.convblock6(x8)# input:8x8x32 output: 8x8x32
        x10 = self.convblock7(torch.sum(torch.stack([x8,x9]), dim=0))# input:8x8x32 output: 8x8x32
        x11 = self.convblock8(torch.sum(torch.stack([x8,x9,x10]), dim=0))# input:8x8x32 output: 8x8x32

        x12 = self.gap(x11)# input:8x8x32 output: 1x1x32
        x13 = self.convblock9(x12)# input:32x32x3 output: ?
        output = x13.view(-1, 10)
        output = F.log_softmax(output,dim=1)

        return output 


