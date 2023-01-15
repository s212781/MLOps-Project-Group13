# from src.features.build_features import MyAwesomeModel as Mymodel
# from src.models import train_model
import dill
import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

data_set = torch.load("data/processed/data_tensor.pt", pickle_module=dill)


train_dataset, test_dataset = torch.utils.data.random_split(data_set, [0.8, 0.2])
