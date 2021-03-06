"""."""
import torch
from torchvision import datasets
from data_transforms import train_transforms
from data_transforms import test_transforms
import os


def get_train_validation_test_data_loaders(data_dir, batch_size, device):
    """."""
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        test_transforms
    )
    # test_dataset = datasets.ImageFolder(
    #     os.path.join(data_dir, 'test'),
    #     test_transforms
    # )
    
    # dataloader arguments
    train_dataloader_args = (
        dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) 
        if device == "cuda" else dict(shuffle=True, batch_size=16)
    )
    test_dataloader_args = (
        dict(shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True) 
        if device == "cuda" else dict(shuffle=True, batch_size=16)
    )

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_dataloader_args)

    # val dataloader
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_dataloader_args)

    # test dataloader
    # test_loader = torch.utils.data.DataLoader(test_dataset, **test_dataloader_args)

    return train_loader, val_loader