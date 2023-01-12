###################################################################
# Script to convert data to something the frameworks can work with#
###################################################################
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import dill
import cv2
import pytest

def mnist(path):
    """ Using dogs dataset """
    
    class MyDataset(Dataset):
        def __init__(self, path):
            lbs = []
            imgs = []
            for c in os.listdir(path):
                img_path = path+c
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img,(100,100))
                
                label = int(img_path[img_path.find('n0')+1:img_path.find('_')])
          
                lbs.append(label)
                imgs.append(img)

            self.images = torch.tensor(imgs).reshape(-1,1,100,100)
            self.labels = torch.tensor(lbs)
       
        def __len__(self):
            return self.images.shape[0]

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]
    
    data_set = MyDataset(path)

    return data_set

if __name__ == "__main__":
    path = "data/external/images/all/"
    data_set = mnist(path)

    output_path = "data/processed/"

    torch.save(data_set, output_path + 'data_tensor.pt', pickle_protocol=True, pickle_module=dill)
