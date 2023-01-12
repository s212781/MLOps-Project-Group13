import os
import pytest
import numpy as np

##TO BE FILLED
N_train = 0
N_test = 0

data_shape = [0, 0, 0]

num_labels = 120

train_data = None
test_data = None

labels = None
##

class DataTest:

    def test_data_len():
        assert len(train_data) == N_train and len(test_data) == N_test

    def test_data_shape():
        for i in train_data:
            assert i.shape == data_shape
        for i in test_data:
            assert i.shape == data_shape

    def test_labels():
        assert np.unique(labels) == num_labels