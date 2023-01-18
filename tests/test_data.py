import os
from torchvision.datasets import ImageFolder

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data


def test_data():
    """
    Test data for model training.

    Checking:
    - image mode
    - label
    - dataset length
    """

    dataset = ImageFolder(_PATH_DATA + "/processed/images")

    for image, label in dataset:
        assert image.mode == "RGB", "Unexpected image mode"
        assert label in dataset.targets, "Unexpected label"

    assert len(dataset) == 20580, "Unexpected dataset length"
