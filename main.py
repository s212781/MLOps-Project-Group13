import torch
from torch import optim, nn
import torchvision.models as models
from src.features.build_features import MyAwesomeModel as Mymodel
from src.models import train_model, predict_model
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from src.data.transforms import train_transform, val_transform
import subprocess
from src.models.deploy_model import deploy
import wandb

#####
# from src.data.make_dataset import MyDataset
from src.data.make_dataset_mnist import MyDataset
####



def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE", device)
    # model = create_model()
    model = Mymodel()
    model.to(device)    

    # batch_size, lr, epochs, num_workers, criterion, optimizer = train_params()

    run = wandb.init(project='MLops13')
    wandb.watch(model, log_freq=100)
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    lr  =  wandb.config.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  
    num_workers = 1

    print("Training...")
    train_dataset, valid_dataset = load_data()
    trainloader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    validloader = DataLoader(valid_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    model = train_model.train(model, trainloader, validloader, criterion, optimizer, epochs)
    
    save_checkpoint(model)
    
    validate(model, 'model_v1_0.pth', batch_size, num_workers, criterion)   
    # return model

def validate(model, model_path, batch_size, num_workers, criterion):
    print("Evaluating...")
    _ , valid_dataset = load_data()
    validloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    model = load_checkpoint(model, model_path)
    model.eval()
                
    # Turn off gradients for validation, will speed up inference
    with torch.no_grad():
        test_loss, accuracy = predict_model.validation(model, validloader, criterion)
    
    print("Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
            "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

def create_model():
    # model = models.resnet152(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 120)
    # this is the given configuration for the 'tiny' model
    model = Mymodel()
    return model

# def train_params():
#     bs = 64
#     lr  = 0.001
#     epochs = 3
#     num_workers = 0
      
#     return bs, lr, epochs, num_workers, criterion, optimizer

def load_data():
    # dataset = ImageFolder('data_mnist/')
    # random_seed = 45
    # torch.manual_seed(random_seed);

    # val_pct = 0.3
    # val_size = int(len(dataset)*val_pct)
    # train_size = len(dataset) - val_size

    # train_ds, val_ds = random_split(dataset, [train_size, val_size])
  
    # train_dataset = MyDataset(train_ds, train_transform())
    # val_dataset = MyDataset(val_ds, val_transform())
    dataset = 'data_mnist'
    train_path = dataset + "/train_"
    test_path = dataset + "/test.npz"
    train_dataset = MyDataset(train_path, train=True)
    val_dataset = MyDataset(test_path, train=False)

    return train_dataset, val_dataset

def load_checkpoint(model, filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def save_checkpoint(model):
    # Giving values but they are not used.
    checkpoint = {'input_size': 1,
              'output_size': 10,
              'state_dict': model.state_dict()}

    torch.save(checkpoint, 'model_v1_0.pth')

def sweep_config():
    sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'test_loss'
		},
    'parameters': {
        'batch_size': {'values': [4, 64, 256]},
        'epochs': {'values': [2, 5, 6]},
        'lr': {'max': 0.1, 'min': 0.0001}
        }
    }
    # Initialize sweep by passing in config. (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='MLops13')
    # wandb.init()
    return sweep_id

# def main():
#     #lets use subprocess to import data
#     # subprocess.run((["dvc pull --remote https://github.com/s212781/MLOps-Project-Group13"]), shell=True)
    
    

#     # model = train(model, batch_size, epochs, num_workers, criterion, optimizer)
    
    

if __name__ == "__main__":
    
    sweep_id = sweep_config()

    wandb.agent(sweep_id, function=train, count=4) 
     

    
  




    
    
    
    








# class ImageClassificationBase(nn.Module):
#     # training step
#     def training_step(self, batch):
#         images, labels = batch

#         if torch.cuda.is_available():
#             images = images.cuda()
#             labels = labels.cuda()        
            
#         out = self(images)
#         loss = F.nll_loss(out, labels)
#         return loss
    
#     # validation step
#     def validation_step(self, batch):
#         images, labels = batch
#         if torch.cuda.is_available():
#             images = images.cuda()
#             labels = labels.cuda()   
        
#         out = self(images)
#         loss = F.nll_loss(out, labels)
#         acc = accuracy(out, labels)
#         return {'val_acc':acc.detach(), 'val_loss':loss.detach()}
    
#     # validation epoch end
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()
#         return {'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()}
        
#     # print result end epoch
#     def epoch_end(self, epoch, result):
#         print("Epoch [{}] : train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result["train_loss"], result["val_loss"], result["val_acc"]))
        
# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# class DogBreedDataset(Dataset):
#     def __init__(self, ds, transform=None):
#         self.ds = ds
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.ds)
    
#     def __getitem__(self, idx):
#         img, label = self.ds[idx]
#         if self.transform:
#             img = self.transform(img)  
#             return img, label

# class DogBreedPretrainedResnet34(ImageClassificationBase):
#     def __init__(self):
#         super().__init__()
#         # torchvision.models.ResNet18_Weights
#         self.network = models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
#         # Replace last layer
#         num_ftrs = self.network.fc.in_features
#         self.network.fc = nn.Sequential(
#             nn.Linear(num_ftrs, 120),
#             nn.LogSoftmax(dim=1)
#         )
        
#     def forward(self, xb):
#         return self.network(xb)

# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']
        

# def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func = torch.optim.Adam):
#     torch.cuda.empty_cache()
#     history = []
#     optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
#     # set up one cycle lr scheduler
#     sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
#     for epoch in range(epochs):
#         steps = 0
#         print_every = 1
#         # Training phase
#         model.train()       
#         train_losses = []
#         lrs = []
#         for batch in tqdm(train_loader):
#             steps += 1
             
#             loss = model.training_step(batch)
#             train_losses.append(loss)
            
#             # calculates gradients
#             loss.backward()
            
#             # check gradient clipping 
#             if grad_clip:
#                 nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
#             # perform gradient descent and modifies the weights
#             optimizer.step()
            
#             # reset the gradients
#             optimizer.zero_grad()
            
#             # record and update lr
#             lrs.append(get_lr(optimizer))
            
#             # modifies the lr value
#             sched.step()
#             if steps % print_every == 0:
#                 with torch.no_grad():
#                     # Model in inference mode, dropout is off
                    
#                     # Validation phase
#                     result = evaluate(model, val_loader)
#                     result['train_loss'] = torch.stack(train_losses).mean().item()
#                     result['lrs'] = lrs
#                     model.epoch_end(epoch, result)
#                     history.append(result)
                
        
#     return history
        
# @torch.no_grad()
# def evaluate(model, val_loader):
#     model.eval()
#     outputs = [model.validation_step(batch) for batch in val_loader]
#     return model.validation_epoch_end(outputs)

# # def predict_single(img, label):
# #     xb = img.unsqueeze(0) # adding extra dimension
# #     xb = to_device(xb, device)
# #     preds = model1(xb)                   # change model object here
# #     predictions = preds[0]
    
# #     max_val, kls = torch.max(predictions, dim=0)
# #     print('Actual :', breeds[label], ' | Predicted :', breeds[kls])
# #     plt.imshow(img.permute(1,2,0))
# #     plt.show()

# if __name__ == "__main__": 
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("DEVICE :: ",device)
#     dataset = ImageFolder('data/processed/images/')
#     model = DogBreedPretrainedResnet34()
#     model.to(device)

#     test_pct = 0.3
#     test_size = int(len(dataset)*test_pct)
#     dataset_size = len(dataset) - test_size

#     val_pct = 0.1
#     val_size = int(dataset_size*val_pct)
#     train_size = dataset_size - val_size

#     train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
#     imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#     img_size = [224, 224]
#     train_transform = transforms.Compose([
#        transforms.Resize((img_size[0], img_size[1])),
#         # transforms.Resize((256, 256)),
#         transforms.RandomCrop(img_size[0], padding=4, padding_mode='reflect'),
#         transforms.RandomHorizontalFlip(p=0.3),
#         transforms.RandomRotation(degrees=30),
#         transforms.ToTensor(),
#     #    transforms.Normalize(*imagenet_stats, inplace=True)
        
#     ])


#     val_transform = transforms.Compose([
#         transforms.Resize((img_size[0], img_size[1])),
#         transforms.ToTensor(),
#     #    transforms.Normalize(*imagenet_stats, inplace=True)
#     ])

#     test_transform = transforms.Compose([
#         transforms.Resize((img_size[0], img_size[1])), 
#         transforms.ToTensor(),
#     #    transforms.Normalize(*imagenet_stats, inplace=True)
#     ])

#     train_dataset = DogBreedDataset(train_ds, train_transform)
#     val_dataset = DogBreedDataset(val_ds, val_transform)
#     test_dataset = DogBreedDataset(test_ds, test_transform)
    
#     batch_size = 64

#     # Create DataLoaders
#     train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True)
#     val_dl = DataLoader(val_dataset, batch_size, num_workers=0, pin_memory=True)
#     test_dl = DataLoader(test_dataset, batch_size, num_workers=0, pin_memory=True)

#     fit_one_cycle(epochs=1, max_lr=0.001, model=model, train_loader=train_dl, val_loader=val_dl)