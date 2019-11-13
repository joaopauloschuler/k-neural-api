# K-CAI NEURAL API [![VERSION](https://img.shields.io/github/v/release/joaopauloschuler/k-neural-api)](https://github.com/joaopauloschuler/k-neural-api/releases)
<img align="right" src="docs/cai.png" height="192">
K-CAI NEURAL API is a Keras based neural network API that will allow you to prototype faster!

This project is a subproject from a bigger and older project called [CAI](https://sourceforge.net/projects/cai/) and is sister to the pascal based [CAI NEURAL API](https://github.com/joaopauloschuler/neural-api/blob/master/README.md).

## Prerequisites
You'll need python and pip.

## Installation
### Via Shell
Installing via shell is very simple:
```
git clone https://github.com/joaopauloschuler/k-neural-api.git k
cd k && pip install .
```
### Installing on Google Colab
Place this on the top of your Google Colab Jupyter Notebook:
```
import os

if not os.path.isdir('k'):
  !git clone https://github.com/joaopauloschuler/k-neural-api.git k
else:
  !cd k && git pull

!cd k && pip install .
```
## Features
* `cai.util.create_image_generator`: this wrapper has extremely well tested default parameters for image classification data augmentation. For you to get a better image classification accuracy might be just a case of replacing your current data augmentation generator by this one. Give it a go!
* `cai.datasets.train_model_on_cifar10`: allows you to quickly train a model on CIFAR-10 dataset. It comes with **K-CAI** default data augmentation settings. See [example](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_10.ipynb).
* `cai.datasets.train_model_on_dataset`: allows you to train a model passed as parameter on a **dataset also passed as parameter**. It comes with **K-CAI** default data augmentation settings making simple to test a given neuronal architecture on multiple datasets. See [example](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_100.ipynb).
* `cai.densenet.simple_densenet`: simple way to create DenseNet models. See [example](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_10.ipynb).

## Documentation
The documentation is under construction. Currently, it's mainly composed by source code examples.

### Examples
Some recommended introductory source code examples are:
* [DenseNet BC L40 with CIFAR-10](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_10.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_10.ipynb) . This example shows how to create a densenet model with `cai.densenet.simple_densenet` and easily train it with `cai.datasets.train_model_on_cifar10`.
* [DenseNet BC L40 with CIFAR-100](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_100.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_100.ipynb) . This example shows how to create a densenet model with `cai.densenet.simple_densenet` and easily train it with `cai.datasets.train_model_on_dataset`.
