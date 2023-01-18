import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "main")))
from main import load_data


def test_data():
    train_dataset, valid_dataset = load_data()
    for image, label in train_dataset:
        with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
            image.shape == [1, 3, 224, 224]
        with pytest.raises(ValueError, match="Expected input to be a 2D tensor"):
            label.shape == [1, 1]

    for image, label in valid_dataset:
        with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
            image.shape == [1, 3, 224, 224]
        with pytest.raises(ValueError, match="Expected input to be a 2D tensor"):
            label.shape == [1, 1]


test_data()
