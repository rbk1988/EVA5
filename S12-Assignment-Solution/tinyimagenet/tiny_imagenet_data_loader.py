"""."""
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import random_split
from data_transforms import train_transforms
from data_transforms import test_transforms
import os

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_train_validation_test_data_loaders(data_dir, batch_size, device):
    """."""
    dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train')
    )
    classes = dataset.classes

    train_split = int(0.7*len(dataset))
    val_split = len(dataset) - train_split
    train_subset, val_subset = random_split(
        dataset,
        (train_split, val_split)
    )

    train_dataset = DatasetFromSubset(
        subset=train_subset,
        transform=train_transforms
    )
    val_dataset = DatasetFromSubset(
        subset=val_subset,
        transform=test_transforms
    )
    # dataloader arguments
    train_dataloader_args = (
        dict(
            shuffle=True,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        ) 
        if device == "cuda" else dict(shuffle=True, batch_size=16)
    )
    test_dataloader_args = (
        dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        ) 
        if device == "cuda" else dict(shuffle=False, batch_size=16)
    )

    # train dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        **train_dataloader_args
    )

    # val dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        **test_dataloader_args
    )

    # test dataloader
    # test_loader = torch.utils.data.DataLoader(test_dataset, **test_dataloader_args)

    return train_loader, val_loader, classes