import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import dill
import cv2

def mnist(path):
    """ Using dogs dataset """
    
    class MyDataset(Dataset):
        def __init__(self, path):
            datas = []
            # First approach is by looking at the xml files for info and then pair with image
            # Loop to access all folders
            #print(len(next(os.walk('dir_name'))[1]))
            # path_anno = path + "annotations/Annotations/"
            # for dirpath, dirnames, filenames in os.walk(path_anno):
            #     # loops every folder
                # datas.append(np.load((path + str(i) + ".npz"), allow_pickle=True))
            # print("this path?", path)
            lbs = []
            for c in os.listdir(path):
                # print("path",c)
                img_path = path+c
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img,(100,100))
                # print(img)
                
                self.imgs = torch.tensor(np.concatenate(img)).reshape(-1,1,100,100)
                label = int(img_path[img_path.find('n0')+1:img_path.find('_')])
                # print("LABEL",int(label))
                # print("TYPE", type(label))
                lbs.append(label)
            self.labels = torch.tensor(lbs)
            # self.imgs = torch.tensor(np.concatenate([c for c in path])).reshape(-1, 1, 28, 28)
            # self.labels = torch.tensor(np.concatenate([c for c in path]))
            
            
            # print(dirpath[dirpath.find('n'):dirpath.find('_')])

        def __len__(self):
            return self.imgs.shape[0]

        def __getitem__(self, idx):
            return self.imgs[idx], self.labels[idx]


    # train_path = r"C:\Users\Thor\Documents\dtu_MLops_answers\S5\M15\Data\data_corrupt\corruptmnist\train_"
    # test_path = r"C:\Users\Thor\Documents\dtu_MLops_answers\S5\M15\Data\data_corrupt\corruptmnist\test.npz"
    # train_path = path + "/train_"
    # test_path = path + "/test.npz"
    
    data = MyDataset(path)
    
    # trainloader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
    # testloader = DataLoader(dataset=test_data, batch_size=bs, shuffle=True)

    """Using normal dataset """
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))])
    # # Download and load the training data
    # trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=ToTensor())
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # # Download and load the test data
    # testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=ToTensor())
    # testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return data

def dir_len():
    path = "data/external/images/Images"
    files = folders = 0

    for dirpath, dirnames, filenames in os.walk(path):
        # print(dirpath)
        # print(dirpath[dirpath.find('n0'):dirpath.find('-')])
        print(dirpath)
        files += len(filenames)
        folders += len(dirnames)

    print("{:,} files, {:,} folders".format(files, folders))

if __name__ == "__main__":
    path = "data/external/images/all/"
    data = mnist(path)
    # input_path = "data/external/"
    output_path = "data/processed/"

    # train_data, test_data = mnist(input_path)
    torch.save(data, output_path + 'data_tensor.pt', pickle_protocol=True, pickle_module=dill)
    # torch.save(test_data, output_path + 'test_tensor.pt', pickle_protocol=True, pickle_module=dill)
