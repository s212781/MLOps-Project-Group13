import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models


_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data

sys.path.append(_PROJECT_ROOT)
from main import create_model


def test_model():
    model = create_model()

    input = torch.zeros((1, 3, 228, 228))
    output = model(input)
    assert output.shape == torch.Size([1, 120])


test_model()
