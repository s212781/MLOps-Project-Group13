DoggieDoggie
==============================
## Introduction
Throughout this project, the aim is to develop a machine learning operations pipeline for a simple dogs breed classification task. The material learnt through DTU's 02476 Machine Learning Operations course will be applied through the course of this porject.

## Dataset
To create a good classifier, a good dataset is needed. The Standford Dogs Dataset found on Kaggle will be used. It includes 20.000 images and features 120 different dog breeds.

## Framework
For the project, the Pytorch Image Models (timm) framework will be used since it is one of the most used computer vision packages. The timm framework is part of fast.ai and is part of the Hugging Face ecosystem. The idea of using this high-level framework is to standardize particular so that attention can be paid to what really matters.

## Models
Since the dataset is made up of images, the optimal model needs to be convolutional Neural Networks. Going into more detail, by utilising the summary of models given in the timm documentation, potential model contenders can be:

- Cross-Stage Partial Networks
- Xception
- Inception-V3

Since, at first glance they seem optimal models for image calssification, with much information to base the code on and look up.


## Other frameworks
Albumentations - Could potentially be used to augment data if training is performed on the framework. This is uncertain since training could take too long given our hardware limititations. 


Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>




