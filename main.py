import os
import torch
import hydra
from torch import optim, nn
import torchvision.models as models
from torchvision.datasets import ImageFolder
from src.data.make_dataset import MyDataset
from src.models import train_model, predict_model
from torch.utils.data import random_split, DataLoader
from src.data.transforms import train_transform, val_transform
from src.features.build_features import MyAwesomeModel as Mymodel


@hydra.main(config_path="config", config_name='default_config.yaml')
def all(config):

    # Calls the fucntion to create model and send it to GPUif exist
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE", device)
    model = create_model()
    model.to(device)

    # Hyperparameters using hydra
    hparams = config.experiment
    TRAIN = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"])

    # Train the model if TRAIN =True 
    if TRAIN:
        print("Training...")
        train_dataset, valid_dataset = load_data()
        trainloader = DataLoader(train_dataset, hparams["batch_size"],
                                shuffle=True, pin_memory=True, num_workers=hparams["num_workers"])
        validloader = DataLoader(valid_dataset, hparams["batch_size"],
                                shuffle=False, pin_memory=True, num_workers=hparams["num_workers"])
        # print(model, trainloader, validloader, criterion, optimizer, hparams["epochs"])
        model= train_model.train(model, trainloader, validloader, criterion, optimizer, hparams["epochs"])

        save_checkpoint(model)

    # Validate the model if TRAIN = False
    if not TRAIN:
        print("Evaluating...")
        _, valid_dataset = load_data()
        validloader = DataLoader(dataset=valid_dataset, batch_size=hparams["batch_size"],
                                shuffle=False, pin_memory=True, num_workers=hparams["num_workers"])

        model = load_checkpoint(model, 'model_v1_0.pth')
        model.eval()

        # Turn off gradients for validation, will speed up inference
        with torch.no_grad():
            test_loss, accuracy = predict_model.validation(
                model, validloader, criterion)

        print("Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
            "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

def create_model():
    #Creates the model
    model = models.resnet152(models.ResNet152_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)

    return model

def load_data():
    #Loads the data from the path
    path_full =os.getcwd()
    path_edit = path_full[:path_full.find("/outputs")]
    dataset = ImageFolder(path_edit + '/data/processed/images')

    random_seed = 45
    torch.manual_seed(random_seed)

    val_pct = 0.3
    val_size = int(len(dataset)*val_pct)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dataset = MyDataset(train_ds, train_transform())
    val_dataset = MyDataset(val_ds, val_transform())

    return train_dataset, val_dataset

def load_checkpoint(model, filepath):
    checkpoint = torch.load(filepath, map_location=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))
    # model = Mymodel()
    model.load_state_dict(checkpoint['state_dict'])

    return model

def save_checkpoint(model):
    # Giving values but they are not used.
    checkpoint = {'input_size': 1,
                  'output_size': 120,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'model_v1_0.pth')

if __name__ == "__main__":
    all()