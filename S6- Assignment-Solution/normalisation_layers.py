"""."""
import torch
import torch.nn as nn
import numpy as np


def get_batch_norm_layer(input_dim, virtual_batch_size,use_ghost_batch_norm = False):
    """Return BatchNorm2d or Ghost BatchNorm2d layer"""
    if use_ghost_batch_norm:
        return GhostBatchNorm2d(
            input_dim = input_dim, 
            virtual_batch_size = virtual_batch_size
        )
    else:
        return nn.BatchNorm2d(
            num_features = input_dim
        )
        

class GhostBatchNorm2d(nn.Module):
    """
        Ghost Batch Normalization
        https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GhostBatchNorm2d, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm2d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)