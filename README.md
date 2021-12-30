# K-CAI NEURAL API [![VERSION](https://img.shields.io/github/v/release/joaopauloschuler/k-neural-api)](https://github.com/joaopauloschuler/k-neural-api/releases)[![DOI](https://zenodo.org/badge/221215803.svg)](https://zenodo.org/badge/latestdoi/221215803)
<img align="right" src="docs/cai.png" height="192">
K-CAI NEURAL API is a Keras based neural network API that will allow you to prototype faster!

This project is a subproject from a bigger and older project called [CAI](https://sourceforge.net/projects/cai/) and is sister to the pascal based [CAI NEURAL API](https://github.com/joaopauloschuler/neural-api/).

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
* A number of new layer types.
* `cai.util.create_image_generator`: this wrapper has extremely well tested default parameters for image classification data augmentation. For you to get a better image classification accuracy might be just a case of replacing your current data augmentation generator by this one. Give it a go!
* `cai.util.create_image_generator_no_augmentation`: image generator for test datasets.
* `cai.densenet.simple_densenet`: simple way to create DenseNet models. See [example](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_10.ipynb).
* `cai.datasets.load_hyperspectral_matlab_image`: downloads (if required) and loads hyperspectral image from a matlab file. This function has been tested with [AVIRIS](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) and ROSIS sensor data stored as a matlab files.
* `cai.models.calculate_heat_map_from_dense_and_avgpool`: calculates a class activation mapping (CAM) inspired on the paper [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150) (see example below).
* `cai.util.show_neuronal_patterns`: creates an array for visualizing first layer neuronal filters/patterns (see example below).
* `cai.gradientascent.run_gradient_ascent_octaves`: allows visualizing patterns recognized by inner neuronal layers. See [example](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_gradient_ascent.ipynb).
* `cai.datasets.save_tfds_in_format`: saves a TensorFlow dataset as image files. Classes are folders.
* `cai.datasets.load_images_from_folders`: practical way to load small datasets into memory. It supports smart resizing, LAB color encoding and bipolar inputs.

## New Layer Types
* `cai.layers.CopyChannels`: copies a subset of the input channels.
* `cai.layers.Negate`: negates (multiplies by -1) the input tensor.
* `cai.layers.ConcatNegation`: concatenates the input with its negation.
* `cai.layers.InterleaveChannels`: interleaves channels stepping according to the number passed as parameter.
* `cai.layers.SumIntoHalfChannels`: divedes channels into 2 halfs and then sums both halfs. This results into an output with the half of the input channels.
* `cai.layers.GlobalAverageMaxPooling2D`: adds both global Average and Max poolings. This layers is known to speed up training.
* `cai.layers.kPointwiseConv2D`: parameter efficient pointwise convolution as shown in the paper [Grouped Pointwise Convolutions Significantly Reduces Parameters in EfficientNet](https://github.com/joaopauloschuler/kEffNet).

## Documentation
The documentation is composed by **examples** and **PyDoc**.

### Examples
Some recommended introductory source code examples are:
* [Simple Image Classification with any Dataset](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/simple_image_classification_with_any_dataset.ipynb): this example shows how to create a model and train it with a dataset passed as parameter. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/simple_image_classification_with_any_dataset.ipynb)
* [DenseNet BC L40 with CIFAR-10](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_10.ipynb): this example shows how to create a densenet model with `cai.densenet.simple_densenet` and easily train it with `cai.datasets.train_model_on_cifar10`. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_10.ipynb)
* [DenseNet BC L40 with CIFAR-100](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_100.ipynb): this example shows how to create a densenet model with `cai.densenet.simple_densenet` and easily train it with `cai.datasets.train_model_on_dataset`. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_densenet_bc_l40_with_cifar_100.ipynb)
* [Experiment your own DenseNet Architecture](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/densenet_with_cifar.ipynb): this example allows you to experiment your own DenseNet settings. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/densenet_with_cifar.ipynb)
* [Gradient Ascent / Deep Dream Example](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_gradient_ascent.ipynb): this example shows how you can quickly display heatmap (CAM), activation maps and first layer filters/patterns.  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/cai_gradient_ascent.ipynb)
* [Heatmap and Activation Map Examples with CIFAR-10](https://github.com/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/activation_map_heatmap_with_cifar10.ipynb): this example shows how you can quickly display heatmap (CAM), activation maps and first layer filters/patterns.
* [kEffNet](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/kEffNet_v1.ipynb): shows how to create and run kEffNet described in the paper [Grouped Pointwise Convolutions Significantly Reduces Parameters in EfficientNet](https://github.com/joaopauloschuler/kEffNet).
* [Saving a TensorFlow dataset into png files](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/k_cai_tfds_example.ipynb).

The following image shows a car (input sample), its heatmap and both added together.


<p><img src="docs/cai-heatmap.png"></img></p>

Heatmaps can be produced following this example:

```
heat_map = cai.models.calculate_heat_map_from_dense_and_avgpool(InputImage, image_class, model, pOutputLayerName='last_conv_layer', pDenseLayerName='dense')
```

These are activation map examples:
<p><img src="docs/cai-activations.png"></img></p>
Activation maps above have been created with a code similar to this:

```
conv_output = cai.models.PartialModelPredict(InputImage, model, 'layer_name', False)
...
activation_maps = cai.util.slice_3d_into_2d(aImage=conv_output[0], NumRows=8, NumCols=8, ForceCellMax=True);
...
plt.imshow(activation_maps, interpolation='nearest', aspect='equal')
```

These are filter examples:

<p><img src="docs/cai-filters.png"></img></p>

Above image has been created with a code similar to this:

```
weights = model.get_layer('layer_name').get_weights()[0]
neuron_patterns = cai.util.show_neuronal_patterns(weights, NumRows = 8, NumCols = 8, ForceCellMax = True)
...
plt.imshow(neuron_patterns, interpolation='nearest', aspect='equal')
```
### Gradient Ascent & Deep Dream
With **cai.gradientascent.run_gradient_ascent_octaves**, you can easily run gradient ascent to create Deep Dream like images:
```
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
pmodel = cai.models.CreatePartialModel(base_model, 'mixed3')
new_img = cai.gradientascent.run_gradient_ascent_octaves(img=original_img, partial_model=pmodel, low_range=-4, high_range=1)
plt.figure(figsize = (16, 16))
plt.imshow(new_img, interpolation='nearest', aspect='equal')
plt.show()
```
<p><img src="docs/park-ga.jpg"></img></p>

Above image was generated from:

<p><img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/park.jpg?raw=true" width=714px></img></p>

### Scientific Research
Experiments done for these papers were made with K-CAI API:
* [Grouped Pointwise Convolutions Significantly Reduces Parameters in EfficientNet](https://github.com/joaopauloschuler/kEffNet).
* [Reliable Deep Learning Plant Leaf Disease Classification Based on Light-Chroma Separated Branches](https://ebooks.iospress.nl/volumearticle/57735).

### PyDoc
After installing K-CAI, you can find documentation with:
```
python -m pydoc cai.datasets
python -m pydoc cai.densenet
python -m pydoc cai.util
python -m pydoc cai.models
```
