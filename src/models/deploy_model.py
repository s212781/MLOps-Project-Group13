import torch
from PIL import Image
from torchvision import transforms


def deploy(modelpath, imagepath):

    model = torch.load(modelpath,map_location=torch.device('cpu'))
    image = Image.open(imagepath)
    transform = transforms.Compose(
        [transforms.Resize([224, 224]), transforms.ToTensor()]
    )
    input = transform(image)
    input = torch.unsqueeze(input, dim=0)
    output = model(input)
    label = torch.argmax(output).item()
    return label
