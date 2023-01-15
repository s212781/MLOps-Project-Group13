###################################################################
# Script to convert data to something the frameworks can work with#
###################################################################
import os

import cv2
import dill
import torch
from convert_label import create_lbl_lst
from torch.utils.data import Dataset


def mnist(path):
    """ Using dogs dataset """
    h = w = 28

    class MyDataset(Dataset):
        def __init__(self, path):
            lbs = []
            lbs2 = []  # sry
            imgs = []
            for c in os.listdir(path):
                img_path = path + c
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (h, w))
                label = int(c[c.find("n0") + 5: c.find("_")])
                # label = int(img_path[img_path.find('n0')+5:img_path.find('_')])

                lbs.append(label)
                imgs.append(img)

            label_dict = create_lbl_lst()

            for i in range(len(lbs)):
                lbs2.append(label_dict.index(lbs[i]))

            self.images = torch.tensor(imgs).reshape(-1, 1, h, w)
            self.labels = torch.tensor(lbs2)

        def __len__(self):
            return self.images.shape[0]

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    data_set = MyDataset(path)

    return data_set


if __name__ == "__main__":
    path = "data/external/images/all_cropped/"
    data_set = mnist(path)

    output_path = "data/processed/"

    torch.save(data_set, output_path + "data_tensor.pt", pickle_protocol=True, pickle_module=dill)
