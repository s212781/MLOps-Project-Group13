import torch
import wandb
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader

from src.features.build_features import MyAwesomeModel as Mymodel
from src.models import predict_model
from src.models import train_model
# from src.features.build_features import MyAwesomeModel as Mymodel
# from src.models import train_model
import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm.notebook import tqdm

class ImageClassificationBase(nn.Module):
    # training step
    def training_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.nll_loss(out, targets)
        return loss
    
    # validation step
    def validation_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.nll_loss(out, targets)
        acc = accuracy(out, targets)
        return {'val_acc':acc.detach(), 'val_loss':loss.detach()}
    
    # validation epoch end
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()}
        
    # print result end epoch
    def epoch_end(self, epoch, result):
        print("Epoch [{}] : train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result["train_loss"], result["val_loss"], result["val_acc"]))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class DogBreedDataset(Dataset):
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

class DogBreedPretrainedResnet34(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_ftrs, 120),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, xb):
        return self.network(xb)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func = torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # set up one cycle lr scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    for epoch in range(epochs):
        steps = 0
        print_every = 1
        # Training phase
        model.train()       
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            steps += 1
            loss = model.training_step(batch)
            train_losses.append(loss)
            
            # calculates gradients
            loss.backward()
            
            # check gradient clipping 
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            # perform gradient descent and modifies the weights
            optimizer.step()
            
            # reset the gradients
            optimizer.zero_grad()
            
            # record and update lr
            lrs.append(get_lr(optimizer))
            
            # modifies the lr value
            sched.step()
            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                
                # Validation phase
                result = evaluate(model, val_loader)
                result['train_loss'] = torch.stack(train_losses).mean().item()
                result['lrs'] = lrs
                model.epoch_end(epoch, result)
                history.append(result)
                
        
    return history
        
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

if __name__ == "__main__": 
    dataset = ImageFolder('data/external/images/Images')
    model = DogBreedPretrainedResnet34()
    test_pct = 0.3
    test_size = int(len(dataset)*test_pct)
    dataset_size = len(dataset) - test_size

    val_pct = 0.1
    val_size = int(dataset_size*val_pct)
    train_size = dataset_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
    #    transforms.Normalize(*imagenet_stats, inplace=True)
        
    ])


    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    #    transforms.Normalize(*imagenet_stats, inplace=True)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
    #    transforms.Normalize(*imagenet_stats, inplace=True)
    ])

    train_dataset = DogBreedDataset(train_ds, train_transform)
    val_dataset = DogBreedDataset(val_ds, val_transform)
    test_dataset = DogBreedDataset(test_ds, test_transform)
    
    batch_size = 64

    # Create DataLoaders
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size*2, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size*2, num_workers=2, pin_memory=True)

    fit_one_cycle(epochs=1, max_lr=0.001, model=model, train_loader=train_dl, val_loader=val_dl)