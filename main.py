import dill
import torch

data_set = torch.load("data/processed/data_tensor.pt", pickle_module=dill)


train_dataset, test_dataset = torch.utils.data.random_split(data_set, [0.8, 0.2])
