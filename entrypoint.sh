#!/bin/sh
dvc pull data_mnist
wandb login c2dcb8a6504297b59ce85d4d01a9d560eb1f0934
# python -u src/data/make_dataset.py preprocess-command --num_samples=10
python3 -u main.py