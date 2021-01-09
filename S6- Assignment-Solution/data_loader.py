"""."""
import torch
from torchvision import datasets
from torchvision import transforms

# Train Phase transformations
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomRotation((-7.0, 7.0)),
        transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
    ]
)
# Test Phase transformations
test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)


def get_train_test_data_loaders(random_seed=1, batch_size= 4096):
    """."""
    SEED = random_seed
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    train_dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    test_dataloader_args = dict(shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **train_dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **test_dataloader_args)

    return train_loader,test_loader
