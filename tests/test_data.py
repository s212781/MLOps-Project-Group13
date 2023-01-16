import pytest

from tests import _PATH_DATA
from src.data.make_dataset import mnist

import os.path


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():

    dataset = mnist(_PATH_DATA)

    with pytest.raises(ValueError, match="Unexpected dataset size"):
        len(dataset) == 20000

    for data, label in dataset:
        with pytest.raises(ValueError, match="Unexpected data shape"):
            data.size == [1, 3, 28, 28]
        with pytest.raises(ValueError, match="Unexpected label"):
            label in dataset.labels
