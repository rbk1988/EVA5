"""."""
import torch
import torchvision
from torchvision import transforms

# Train Phase transformations
train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
        transforms.RandomRotation(10),     #Rotates the image to a specified angel
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)
# Test Phase transformations
test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def get_train_test_data_loaders(random_seed=1, batch_size= 128):
    """."""
    SEED = random_seed
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    train = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=train_transforms
    )
    test = torchvision.datasets.CIFAR10(
        root='./data', 
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
