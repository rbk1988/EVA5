"""."""
import torch
import torchvision
from dataset import Cifar10_Dataset
from image_augmentations import train_transforms
from image_augmentations import test_transforms


def get_train_test_data_loaders(random_seed=1, batch_size= 128):
    """."""
    SEED = random_seed
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    train = Cifar10_Dataset(
        root="~/data/cifar10", 
        train=True,
        download=True, 
        transform=train_transforms
    )
    test = Cifar10_Dataset(
        root="~/data/cifar10", 
        train=False,
        download=True, 
        transform=test_transforms
    )

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    train_dataloader_args = (
        dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) 
        if cuda else dict(shuffle=True, batch_size=16)
    )
    test_dataloader_args = (
        dict(shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True) 
        if cuda else dict(shuffle=True, batch_size=16)
    )

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **train_dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **test_dataloader_args)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader,test_loader, classes
