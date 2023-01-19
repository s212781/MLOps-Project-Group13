import torch
import hydra
from torch import optim, nn
import torchvision.models as models
from src.features.build_features import MyAwesomeModel as Mymodel
from src.models import train_model, predict_model
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from src.data.make_dataset import MyDataset
from src.data.transforms import train_transform, val_transform


def train(model, batch_size, epochs, num_workers, criterion, optimizer):
    print("Training...")
    train_dataset, valid_dataset = load_data()
    trainloader = DataLoader(train_dataset, batch_size,
                             shuffle=True, pin_memory=True, num_workers=num_workers)
    validloader = DataLoader(valid_dataset, batch_size,
                             shuffle=False, pin_memory=True, num_workers=num_workers)

    train_model.train(model, trainloader, validloader, criterion, optimizer, epochs)


def validate(model, model_path, batch_size, num_workers, criterion):
    print("Evaluating...")
    _, valid_dataset = load_data()
    validloader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                             shuffle=False, pin_memory=True, num_workers=num_workers)

    model = load_checkpoint(model, model_path)
    model.eval()

    # Turn off gradients for validation, will speed up inference
    with torch.no_grad():
        test_loss, accuracy = predict_model.validation(
            model, validloader, criterion)

    print("Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))


def create_model():
    model = models.resnet152(models.ResNet152_Weights.DEFAULT)
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 120)

    return model

@hydra.main(config_path="config", config_name='default_config.yaml')
def train_params(config):
    hparams = config.experiment
    bs = hparams["batch_size"]
    lr = hparams["learning_rate"]
    epochs = hparams["epochs"]
    num_workers = hparams["num_workers"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=hparams["momentum"])
    return bs, lr, epochs, num_workers, criterion, optimizer


def load_data():
    dataset = ImageFolder('data/processed/images')

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE", device)
    model = create_model()
    model.to(device)

    batch_size, lr, epochs, num_workers, criterion, optimizer = train_params()

    model = train(model, batch_size, epochs, num_workers, criterion, optimizer)
    # validate(model, 'model_v1_0.pth', batch_size, num_workers, criterion)

    save_checkpoint(model)