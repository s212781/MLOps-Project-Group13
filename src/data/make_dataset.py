# ##################################################################
# # Script to convert data to something the frameworks can work with#
# ###################################################################
import numpy as np
from torch.utils.data import Dataset

""" Using corrupted dataset """


class MyDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
            return img, label
