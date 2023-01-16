import torch
import pytest

from src.models import MyAwesomeModel


def test_models():

    model = MyAwesomeModel()
    input = torch.zeros((1, 3, 28, 28))
    output = model(input)

    with pytest.raises(ValueError, "Unexpected model"):
        input.size == output.size
