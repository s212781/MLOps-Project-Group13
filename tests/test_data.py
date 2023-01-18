import os
from torchvision.datasets import ImageFolder


def test_data():
    """
    Test data for model training.

    Checking:
    - image mode
    - label
    - dataset length
    """

    currentfolder = os.getcwd()
    dataset = ImageFolder(currentfolder + "/data/processed/images")

    for image, label in dataset:
        assert image.mode == "RGB", "Unexpected image mode"
        assert label in dataset.targets, "Unexpected label"

    assert len(dataset) == 20580, "Unexpected dataset length"
