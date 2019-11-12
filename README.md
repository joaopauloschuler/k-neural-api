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

## Documentation

### Examples
Some recommended introductory source code examples are:
* [DenseNet BC L40 with CIFAR-10](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_10.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_10.ipynb)
