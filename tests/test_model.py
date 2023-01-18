import torch
import torch.nn as nn
import torchvision.models as models


def test_model():
    model = models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)

    input = torch.zeros((1, 3, 228, 228))
    output = model(input)
    assert output.shape == torch.Size([1, 120])
