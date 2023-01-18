import torchvision.transforms as transforms


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
        ]
    )


def val_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
