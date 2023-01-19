#!/bin/sh
dvc pull data_lite
wandb login c2dcb8a6504297b59ce85d4d01a9d560eb1f0934
# python -u src/data/make_dataset.py preprocess-command --num_samples=10
sudo apt-get -y install curl
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
python3 install_gpu_driver.py
python3 -u main.py