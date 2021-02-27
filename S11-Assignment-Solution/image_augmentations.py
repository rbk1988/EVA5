import albumentations as A
from albumentations.pytorch import ToTensorV2

# Train Phase transformations
train_transforms = A.Compose(
    [
        A.PadIfNeeded(min_height=40, min_width=40),
        A.RandomCrop(height=32, width=32, p=1),
        A.HorizontalFlip(p=0.5),
        A.Cutout(num_holes=3,  max_h_size=8, max_w_size=8, p=0.5),
        A.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
        ToTensorV2(),
    ]
)
# Test Phase transformations
test_transforms = A.Compose(
    [
        A.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
        ToTensorV2(),
    ]
)