import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

from src.data.make_dataset import MyDataset
from src.data.transforms import train_transform, val_transform


def load_dataset():
    dataset = ImageFolder("data/processed/images")

    random_seed = 45
    torch.manual_seed(random_seed)

    val_pct = 0.3
    val_size = int(len(dataset) * val_pct)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dataset = MyDataset(train_ds, train_transform())
    val_dataset = MyDataset(val_ds, val_transform())

    return train_dataset, val_dataset
