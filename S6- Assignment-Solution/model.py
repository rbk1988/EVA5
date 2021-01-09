"""Define the class file for the model.."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from normalisation_layers import get_batch_norm_layer


class Net(nn.Module):
    """Define the model layers."""
    def __init__(self,use_gbn_flag= False, virtual_batch_size=128) :
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1, bias=False),
            get_batch_norm_layer(
                input_dim=10, 
                virtual_batch_size=virtual_batch_size,
                use_ghost_batch_norm = use_gbn_flag
            ),
            nn.ReLU()
        ) #input 28x28x1 output 28x28x10 RF 3x3
        self.convblock2 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1, bias=False),
            get_batch_norm_layer(
                input_dim=10, 
                virtual_batch_size=virtual_batch_size,
                use_ghost_batch_norm = use_gbn_flag
            ),
            nn.ReLU()
        ) #input 28x28x10 output 28x28x10 RF 5x5
        self.pool1 = nn.MaxPool2d(2, 2) #input 28x28x10 output 14x14x10 RF 10x10
        self.convblock3 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1, bias=False),
            get_batch_norm_layer(
                input_dim=10, 
                virtual_batch_size=virtual_batch_size,
                use_ghost_batch_norm = use_gbn_flag
            ),
            nn.ReLU()
        ) #input 14x14x10 output 14x14x10 RF 12x12
        self.convblock4 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1, bias=False),
            get_batch_norm_layer(
                input_dim=10, 
                virtual_batch_size=virtual_batch_size,
                use_ghost_batch_norm = use_gbn_flag
            ),
            nn.ReLU()
        ) #input 14x14x10 output 14x14x10 RF 14x14
        self.convblock5 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1, bias=False),
            get_batch_norm_layer(
                input_dim=10, 
                virtual_batch_size=virtual_batch_size,
                use_ghost_batch_norm = use_gbn_flag
            ),
            nn.ReLU()
        ) #input 14x14x10 output 14x14x10 RF 16x16
        self.pool2 = nn.MaxPool2d(2, 2) #input 14x14x10 output 7x7x10 RF 32x32
        self.convblock6 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding =1,bias=False),
            get_batch_norm_layer(
                input_dim=16, 
                virtual_batch_size=virtual_batch_size,
                use_ghost_batch_norm = use_gbn_flag
            ),
            nn.ReLU()
        ) #input 7x7x10 output 7x7x16 RF 34x34
        self.convblock7 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding =0, bias=False),
            get_batch_norm_layer(
                input_dim=16, 
                virtual_batch_size=virtual_batch_size,
                use_ghost_batch_norm = use_gbn_flag
            ),
            nn.ReLU()
        ) #input 7x7x16 output 5x5x16 RF 36x36
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) #input 5x5x16 output 1x1x16 RF ?
        self.convblock8 = nn.Sequential(
            nn.Conv2d(16, 10, 1, bias=False)
        )  #input 1x1x16 output 1x1x10 RF ?

    def forward(self, x):
        """Define the feedforward function.."""
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)