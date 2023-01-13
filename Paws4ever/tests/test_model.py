import pytest
import torch
from src.models import train_model

##TO BE FILLED
model_input_shape = 1,1,1,1
model_output_shape = 1,1

model = train_model()

class TestModel:

    def test_model():
        assert model(torch.rand(model_input_shape)) == (model_output_shape), "Model input and output shape do not match"