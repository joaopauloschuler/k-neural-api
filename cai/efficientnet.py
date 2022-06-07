"""EfficientNet models for Keras.
# Reference paper
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]
  (https://arxiv.org/abs/1905.11946) (ICML 2019)
# Reference implementation
- [TensorFlow]
  (https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
  
COPYRIGHT

Copyright (c) 2016 - 2018, the respective contributors.
All rights reserved.
Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.
The initial code of this file came from 
https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
(the Keras repository), hence, for author information regarding commits
that occured earlier than the first commit in the present repository,
please see the original Keras repository.
The original file from above link was modified. Modifications can be tracked via 
git commits at 
https://github.com/joaopauloschuler/k-neural-api/blob/master/cai/efficientnet.py.

LICENSE

The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cai.util
import cai.models
import cai.layers

import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import utils
from tensorflow.keras.applications import imagenet_utils
import tensorflow
from copy import deepcopy

def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 1 # 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

SHALLOW_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 1, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 1, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 1, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

TWO_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def swish(x):
    """Swish activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The Swish activation: `x * sigmoid(x)`.
    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if backend.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return backend.tf.nn.swish(x)
        except AttributeError:
            pass

    return x * backend.sigmoid(x)


def block(inputs, activation_fn=swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):
    """A mobile inverted residual block.
    # Arguments
        inputs: input tensor.
        activation_fn: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=name + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, kernel_size),
                                 name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=strides,
                               padding=conv_pad,
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation_fn, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se = layers.Reshape((filters, 1, 1), name=name + 'se_reshape')(se)
        else:
            se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = layers.Conv2D(filters_se, 1,
                           padding='same',
                           activation=activation_fn,
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_reduce')(se)
        se = layers.Conv2D(filters, 1,
                           padding='same',
                           activation='sigmoid',
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_expand')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = layers.Conv2D(filters_out, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = layers.Dropout(drop_rate,
                               noise_shape=(None, 1, 1, 1),
                               name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')
    return x

def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation_fn=swish,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation_fn: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid input shape.
    """

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = img_input
    x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                             name='stem_conv_pad')(x)
    x = layers.Conv2D(round_filters(32), 3,
                      strides=2,
                      padding='valid',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation_fn, name='stem_activation')(x)

    # Build blocks
    from copy import deepcopy
    blocks_args = deepcopy(blocks_args)

    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, activation_fn, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1

    # Build top
    x = layers.Conv2D(round_filters(1280), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation_fn, name='top_activation')(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    return model

def EfficientNetB0(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.2,
                   drop_connect_rate=0.2,
                   **kwargs):
    return EfficientNet(1.0, 1.0, 224,
                        model_name='efficientnet-b0',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        **kwargs)


def EfficientNetB1(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.2,
                   drop_connect_rate=0.2,
                   **kwargs):
    return EfficientNet(1.0, 1.1, 240,
                        model_name='efficientnet-b1',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        **kwargs)


def EfficientNetB2(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.3,
                   drop_connect_rate=0.2,
                   **kwargs):
    return EfficientNet(1.1, 1.2, 260,
                        model_name='efficientnet-b2',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        **kwargs)


def EfficientNetB3(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.3,
                   drop_connect_rate=0.2,
                   **kwargs):
    return EfficientNet(1.2, 1.4, 300,
                        model_name='efficientnet-b3',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        **kwargs)


def EfficientNetB4(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.4,
                   drop_connect_rate=0.2,
                   **kwargs):
    return EfficientNet(1.4, 1.8, 380,
                        model_name='efficientnet-b4',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        **kwargs)


def EfficientNetB5(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.4,
                   drop_connect_rate=0.2,
                   **kwargs):
    return EfficientNet(1.6, 2.2, 456,
                        model_name='efficientnet-b5',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        **kwargs)


def EfficientNetB6(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.5,
                   drop_connect_rate=0.2,
                   **kwargs):
    return EfficientNet(1.8, 2.6, 528,
                        model_name='efficientnet-b6',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        **kwargs)


def EfficientNetB7(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.5,
                   drop_connect_rate=0.2,
                   **kwargs):
    return EfficientNet(2.0, 3.1, 600,
                        model_name='efficientnet-b7',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        **kwargs)


def kblock(inputs, activation_fn=swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True, kType=1,
          dropout_all_blocks=False):
    """A mobile inverted residual block.
    # Arguments
        inputs: input tensor.
        activation_fn: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        #x = layers.Conv2D(filters, 1,
        #                 padding='same',
        #                  use_bias=False,
        #                  kernel_initializer=CONV_KERNEL_INITIALIZER,
        #                  name=name + 'expand_conv')(inputs)
        #x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        #x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
        x = cai.layers.kPointwiseConv2D(last_tensor=inputs, filters=filters, channel_axis=bn_axis, name=name+'expand', activation=activation_fn, has_batch_norm=True, use_bias=False, kType=kType)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, kernel_size),
                                 name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=strides,
                               padding=conv_pad,
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation_fn, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se = layers.Reshape((filters, 1, 1), name=name + 'se_reshape')(se)
        else:
            se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        #se = layers.Conv2D(filters_se, 1,
        #                   padding='same',
        #                   activation=activation_fn,
        #                   kernel_initializer=CONV_KERNEL_INITIALIZER,
        #                   name=name + 'se_reduce')(se)
        se = cai.layers.kPointwiseConv2D(last_tensor=se, filters=filters_se, channel_axis=bn_axis, name=name+'se_reduce', activation=activation_fn, has_batch_norm=False, use_bias=True, kType=kType)
        #se = layers.Conv2D(filters, 1,
        #                   padding='same',
        #                   activation='sigmoid',
        #                   kernel_initializer=CONV_KERNEL_INITIALIZER,
        #                   name=name + 'se_expand')(se)
        se = cai.layers.kPointwiseConv2D(last_tensor=se, filters=filters, channel_axis=bn_axis, name=name+'se_expand', activation='sigmoid', has_batch_norm=False, use_bias=True, kType=kType)
        x = layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    #x = layers.Conv2D(filters_out, 1,
    #                  padding='same',
    #                  use_bias=False,
    #                  kernel_initializer=CONV_KERNEL_INITIALIZER,
    #                  name=name + 'project_conv')(x)
    # x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    x = cai.layers.kPointwiseConv2D(last_tensor=x, filters=filters_out, channel_axis=bn_axis, name=name+'project_conv', activation=None, has_batch_norm=True, use_bias=False, kType=kType)

    if (drop_rate > 0)  and (dropout_all_blocks):
        x = layers.Dropout(drop_rate,
                noise_shape=(None, 1, 1, 1),
                name=name + 'drop')(x)

    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if (drop_rate > 0)  and (not dropout_all_blocks):
            x = layers.Dropout(drop_rate,
                               noise_shape=(None, 1, 1, 1),
                               name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')
    return x

def kblockLastName(drop_rate=0., name='',
          filters_in=32, filters_out=16, strides=1,
          id_skip=True,
          dropout_all_blocks=False):
    last_name = name + 'project_conv'

    if (drop_rate > 0)  and (dropout_all_blocks):
        last_name = name + 'drop'

    if (id_skip is True and strides == 1 and filters_in == filters_out):
        last_name = name + 'add'
    return last_name

def kEfficientNet(
        width_coefficient,
        depth_coefficient,
        skip_stride_cnt=-1,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation_fn=swish,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        model_name='efficientnet',
        include_top=True,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        kType=2,
        concat_paths=True,
        dropout_all_blocks=False,
        name_prefix='k_',
        **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        skip_stride_cnt: number of layers to skip stride. This parameter is used with smalll images such as CIFAR-10.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation_fn: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid input shape.
    """

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    if isinstance(kType, (int)):
        kTypeList = [kType]
    else:
        kTypeList = kType
    
    # Build stem
    x = img_input
    x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                             name=name_prefix+'stem_conv_pad')(x)
    first_stride = 1 if skip_stride_cnt >= 0 else 2
    x = layers.Conv2D(round_filters(32), 3,
                      strides=first_stride,
                      padding='valid',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name_prefix+'stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name_prefix+'stem_bn')(x)
    x = layers.Activation(activation_fn, name=name_prefix+'stem_activation')(x)

    root_layer = x
    output_layers = []
    path_cnt = 0
    for kType in kTypeList:
        x = root_layer
        blocks_args_cp = deepcopy(blocks_args)
        b = 0
        blocks = float(sum(args['repeats'] for args in blocks_args_cp))
        #only the first branch can backpropagate to the input.
        #if path_cnt>0:
        #    x = keras.layers.Lambda(lambda x: tensorflow.stop_gradient(x))(x)
        for (i, args) in enumerate(blocks_args_cp):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = round_filters(args['filters_in'])
            args['filters_out'] = round_filters(args['filters_out'])

            for j in range(round_repeats(args.pop('repeats'))):
                #should skip the stride
                if (skip_stride_cnt > i) and (j == 0) and (args['strides'] > 1):
                    args['strides'] = 1
                # The first block needs to take care of stride and filter size increase.
                if (j > 0):
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                x = kblock(x, activation_fn, drop_connect_rate * b / blocks,
                          name=name_prefix+'block{}{}_'.format(i + 1, chr(j + 97))+'_'+str(path_cnt), **args,
                          kType=kType, dropout_all_blocks=dropout_all_blocks)
                b += 1
        if (len(kTypeList)>1):
            x = layers.Activation('relu', name=name_prefix+'end_relu'+'_'+str(path_cnt))(x)
        output_layers.append(x)
        path_cnt = path_cnt +1
        
    if (len(output_layers)==1):
        x = output_layers[0]
    else:
        if concat_paths:
            x = keras.layers.Concatenate(axis=bn_axis, name=name_prefix+'global_concat')(output_layers)
        else:
            x = keras.layers.add(output_layers, name=name_prefix+'global_add')

    # Build top
    #x = layers.Conv2D(round_filters(1280), 1,
    #                  padding='same',
    #                  use_bias=False,
    #                  kernel_initializer=CONV_KERNEL_INITIALIZER,
    #                  name='top_conv')(x)
    #x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    #x = layers.Activation(activation_fn, name='top_activation')(x)
    x = cai.layers.kPointwiseConv2D(last_tensor=x, filters=round_filters(1280), channel_axis=bn_axis, name=name_prefix+'top_conv', activation=None, has_batch_norm=True, use_bias=False, kType=kType)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name=name_prefix+'avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name=name_prefix+'max_pool')(x)
    elif pooling == 'avgmax':
        x = cai.layers.GlobalAverageMaxPooling2D(x, name=name_prefix+'avgmax_pool')

    if include_top:
        if (dropout_rate > 0):
            x = layers.Dropout(dropout_rate, name=name_prefix+'top_dropout')(x)
        x = layers.Dense(classes,
            activation='softmax', # 'softmax'
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            name=name_prefix+'probs')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    return model

def AddkEfficientNetPath(
        last_tensor,
        width_coefficient,
        depth_coefficient,
        skip_stride_cnt=-1,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation_fn=swish,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        name_prefix='pre_',
        pooling='avg',
        include_dense=True,
        classes=1000,
        kType=2,
        concat_paths=True,
        dropout_all_blocks=False,
        **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        skip_stride_cnt: number of layers to skip stride. Good for smalll images.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation_fn: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid input shape.
    """

    x=last_tensor
    bn_axis = 3

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    if isinstance(kType, (int)):
        kTypeList = [kType]
    else:
        kTypeList = kType
    
    root_layer = x
    output_layers = []
    path_cnt = 0
    for kType in kTypeList:
        x = root_layer
        blocks_args_cp = deepcopy(blocks_args)
        b = 0
        blocks = float(sum(args['repeats'] for args in blocks_args_cp))

        for (i, args) in enumerate(blocks_args_cp):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = round_filters(args['filters_in'])
            args['filters_out'] = round_filters(args['filters_out'])

            for j in range(round_repeats(args.pop('repeats'))):
                #should skip the stride
                if (skip_stride_cnt > i) and (j == 0) and (args['strides'] > 1):
                    args['strides'] = 1
                # The first block needs to take care of stride and filter size increase.
                if (j > 0):
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                x = kblock(x, activation_fn, drop_connect_rate * b / blocks,
                          name=name_prefix+'block{}{}_'.format(i + 1, chr(j + 97))+'_'+str(path_cnt), **args,
                          kType=kType, dropout_all_blocks=dropout_all_blocks)
                b += 1
        if (len(kTypeList)>1):
            x = layers.Activation('relu', name=name_prefix+'end_relu'+'_'+str(path_cnt))(x)
        output_layers.append(x)
        path_cnt = path_cnt +1
        
    if (len(output_layers)==1):
        x = output_layers[0]
    else:
        if concat_paths:
            x = keras.layers.Concatenate(axis=bn_axis, name=name_prefix+'global_concat')(output_layers)
        else:
            x = keras.layers.add(output_layers, name=name_prefix+'global_add')

    x = cai.layers.kPointwiseConv2D(last_tensor=x, filters=round_filters(1280), channel_axis=bn_axis, name=name_prefix+'top_conv', activation=None, has_batch_norm=True, use_bias=False, kType=kType)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name=name_prefix+'avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name=name_prefix+'max_pool')(x)
    elif pooling == 'avgmax':
        x = cai.layers.GlobalAverageMaxPooling2D(x, name=name_prefix+'avgmax_pool')

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=name_prefix+'top_dropout')(x)
    
    if include_dense:
        x = layers.Dense(classes,
                activation='relu',
                kernel_initializer=DENSE_KERNEL_INITIALIZER,
                name=name_prefix+'probs')(x)
    return x # end of AddkEfficientNetPath

def AddkEfficientNetParallelBlocks(
        last_tensor,
        existing_model,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        skip_stride_cnt=-1,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation_fn=swish,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        name_prefix='pre2_',
        existing_name_prefix='k_',
        pooling='avg',
        include_dense=True,
        classes=1000,
        kType=2,
        concat_paths=True,
        dropout_all_blocks=False,
        **kwargs):

    x=last_tensor
    bn_axis = 3

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    if isinstance(kType, (int)):
        kTypeList = [kType]
    else:
        kTypeList = kType
    
    root_layer = x
    output_layers = []
    path_cnt = 0
    for kType in kTypeList:
        x = root_layer
        blocks_args_cp = deepcopy(blocks_args)
        b = 0
        blocks = float(sum(args['repeats'] for args in blocks_args_cp))

        for (i, args) in enumerate(blocks_args_cp):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = round_filters(args['filters_in'])
            args['filters_out'] = round_filters(args['filters_out'])

            for j in range(round_repeats(args.pop('repeats'))):
                #should skip the stride
                if (skip_stride_cnt > i) and (j == 0) and (args['strides'] > 1):
                    args['strides'] = 1
                # The first block needs to take care of stride and filter size increase.
                if (j > 0):
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                x = kblock(x, activation_fn, drop_connect_rate * b / blocks,
                          name=name_prefix+'block{}{}_'.format(i + 1, chr(j + 97))+'_'+str(path_cnt), **args,
                          kType=kType, dropout_all_blocks=dropout_all_blocks)
                b += 1
            prev_layer_name = x.name
            other_layer_name = prev_layer_name.replace(name_prefix,  existing_name_prefix).split('/')[0]
            # print('[',other_layer_name,']')
            # existing_model.summary()
            other_layer = existing_model.get_layer(other_layer_name).output
            #adds both paths.
            x = layers.add([x, other_layer],  name=name_prefix+'add_'+str(i)+'_'+str(path_cnt))
        
        if (len(kTypeList)>1):
            x = layers.Activation('relu', name=name_prefix+'end_relu'+'_'+str(path_cnt))(x)
        output_layers.append(x)
        path_cnt = path_cnt +1
        
    if (len(output_layers)==1):
        x = output_layers[0]
    else:
        if concat_paths:
            x = keras.layers.Concatenate(axis=bn_axis, name=name_prefix+'global_concat')(output_layers)
        else:
            x = keras.layers.add(output_layers, name=name_prefix+'global_add')

    x = cai.layers.kPointwiseConv2D(last_tensor=x, filters=round_filters(1280), channel_axis=bn_axis, name=name_prefix+'top_conv', activation=None, has_batch_norm=True, use_bias=False, kType=kType)

    prev_layer_name = x.name
    other_layer_name = prev_layer_name.replace(name_prefix, existing_name_prefix).split('/')[0]
    other_layer = existing_model.get_layer(other_layer_name).output
    #adds both paths.
    x = layers.add([x, other_layer], name=name_prefix+'_final_add')

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name=name_prefix+'avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name=name_prefix+'max_pool')(x)
    elif pooling == 'avgmax':
        x = cai.layers.GlobalAverageMaxPooling2D(x, name=name_prefix+'avgmax_pool')

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=name_prefix+'top_dropout')(x)
    
    if include_dense:
        x = layers.Dense(classes,
                activation='softmax',
                kernel_initializer=DENSE_KERNEL_INITIALIZER,
                name=name_prefix+'probs')(x)
    return x # AddkEfficientNetParallelBlocks


def AddkEfficientNetOtherBlocks(
        last_tensor,
        existing_model,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        skip_stride_cnt=-1,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation_fn=swish,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        name_prefix='pre2_',
        existing_name_prefix='k_',
        pooling='avg',
        include_dense=True,
        classes=1000,
        kType=2,
        concat_paths=True,
        dropout_all_blocks=False,
        dropout_extra=False, 
        **kwargs):

    x=last_tensor
    bn_axis = 3

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    if isinstance(kType, (int)):
        kTypeList = [kType]
    else:
        kTypeList = kType
    
    root_layer = x
    output_layers = []
    path_cnt = 0
    for kType in kTypeList:
        x = root_layer
        blocks_args_cp = deepcopy(blocks_args)
        b = 0
        blocks = float(sum(args['repeats'] for args in blocks_args_cp))

        for (i, args) in enumerate(blocks_args_cp):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = round_filters(args['filters_in'])
            args['filters_out'] = round_filters(args['filters_out'])

            for j in range(round_repeats(args.pop('repeats'))):
                #should skip the stride
                if (skip_stride_cnt > i) and (j == 0) and (args['strides'] > 1):
                    args['strides'] = 1
                # The first block needs to take care of stride and filter size increase.
                if (j > 0):
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                block_dropout_rate = drop_connect_rate * b / blocks
                layer_name = name_prefix+'block{}{}_'.format(i + 1, chr(j + 97))+'_'+str(path_cnt)
                x = kblock(x, activation_fn, block_dropout_rate,
                    name=layer_name, **args,
                    kType=kType, dropout_all_blocks=dropout_all_blocks)
                prev_layer_name = x.name
                other_layer_name = prev_layer_name.replace(name_prefix, existing_name_prefix).split('/')[0]
                # print('[',other_layer_name,']')
                # existing_model.summary()
                other_layer = existing_model.get_layer(other_layer_name).output
                #adds both paths.
                if (dropout_extra):
                    x = layers.Dropout(block_dropout_rate,
                        noise_shape=(None, 1, 1, 1),
                        name=layer_name + '_extra_drop')(x)
                x = layers.add([x, other_layer], name=name_prefix+'add_'+str(i)+'_'+str(path_cnt))
                b += 1
        
        if (len(kTypeList)>1):
            x = layers.Activation('relu', name=name_prefix+'end_relu'+'_'+str(path_cnt))(x)
        output_layers.append(x)
        path_cnt = path_cnt +1
        
    if (len(output_layers)==1):
        x = output_layers[0]
    else:
        if concat_paths:
            x = keras.layers.Concatenate(axis=bn_axis, name=name_prefix+'global_concat')(output_layers)
        else:
            x = keras.layers.add(output_layers, name=name_prefix+'global_add')

    x = cai.layers.kPointwiseConv2D(last_tensor=x, filters=round_filters(1280), channel_axis=bn_axis, name=name_prefix+'top_conv', activation=None, has_batch_norm=True, use_bias=False, kType=kType)

    prev_layer_name = x.name
    other_layer_name = prev_layer_name.replace(name_prefix,  existing_name_prefix).split('/')[0]
    other_layer = existing_model.get_layer(other_layer_name).output
    #adds both paths.
    x = layers.add([x, other_layer],  name=name_prefix+'_final_add')

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name=name_prefix+'avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name=name_prefix+'max_pool')(x)
    elif pooling == 'avgmax':
        x = cai.layers.GlobalAverageMaxPooling2D(x, name=name_prefix+'avgmax_pool')

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=name_prefix+'top_dropout')(x)
    
    if include_dense:
        x = layers.Dense(classes,
                activation='softmax',
                kernel_initializer=DENSE_KERNEL_INITIALIZER,
                name=name_prefix+'probs')(x)
    return x # AddkEfficientNetOtherBlocks
    
    
def kEfficientNetS(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling='avg',
                   classes=1000,
                   kType=2,
                   dropout_rate=0.2,
                   drop_connect_rate=0.2,
                   skip_stride_cnt=-1,
                   activation_fn=swish,
                   dropout_all_blocks=False,
                   **kwargs):
    return kEfficientNet(1.0, 1.0, skip_stride_cnt=skip_stride_cnt, # 224,
                        model_name='kEffNet-s',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        activation_fn=activation_fn,
                        blocks_args=SHALLOW_BLOCKS_ARGS,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)

def kEfficientNetB0(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling='avg',
                   classes=1000,
                   kType=2,
                   dropout_rate=0.2,
                   drop_connect_rate=0.2,
                   skip_stride_cnt=-1,
                   activation_fn=swish,
                   dropout_all_blocks=False,
                   **kwargs):
    return kEfficientNet(1.0, 1.0, skip_stride_cnt=skip_stride_cnt, # 224,
                        model_name='kEffNet-b0',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        activation_fn=activation_fn,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)

def kEfficientNetB1(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling='avg',
                   classes=1000,
                   kType=2,
                   dropout_rate=0.2,
                   drop_connect_rate=0.2,
                   skip_stride_cnt=-1, 
                   activation_fn=swish,
                   dropout_all_blocks=False,
                   **kwargs):
    return kEfficientNet(1.0, 1.1, skip_stride_cnt=skip_stride_cnt, #240,
                        model_name='kEffNet-b1',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        activation_fn=activation_fn,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)


def kEfficientNetB2(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling='avg',
                   classes=1000,
                   kType=2,
                   dropout_rate=0.3,
                   drop_connect_rate=0.2,
                   skip_stride_cnt=-1, 
                   activation_fn=swish,
                   dropout_all_blocks=False,
                   **kwargs):
    return kEfficientNet(1.1, 1.2, skip_stride_cnt=skip_stride_cnt, #260,
                        model_name='kEffNet-b2',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        activation_fn=activation_fn,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)


def kEfficientNetB3(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling='avg',
                   classes=1000,
                   kType=2,
                   dropout_rate=0.3,
                   drop_connect_rate=0.2,
                   skip_stride_cnt=-1, 
                   activation_fn=swish,
                   dropout_all_blocks=False,
                   **kwargs):
    return kEfficientNet(1.2, 1.4, skip_stride_cnt=skip_stride_cnt, #300,
                        model_name='kEffNet-b3',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        activation_fn=activation_fn,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)


def kEfficientNetB4(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling='avg',
                   classes=1000,
                   kType=2,
                   dropout_rate=0.4,
                   drop_connect_rate=0.2,
                   skip_stride_cnt=-1, 
                   activation_fn=swish,
                   dropout_all_blocks=False,
                   **kwargs):
    return kEfficientNet(1.4, 1.8, skip_stride_cnt=skip_stride_cnt, #380,
                        model_name='kEffNet-b4',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        activation_fn=activation_fn,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)


def kEfficientNetB5(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling='avg',
                   classes=1000,
                   kType=2,
                   dropout_rate=0.4,
                   drop_connect_rate=0.2,
                   skip_stride_cnt=-1, 
                   activation_fn=swish,
                   dropout_all_blocks=False,
                   **kwargs):
    return kEfficientNet(1.6, 2.2, skip_stride_cnt=skip_stride_cnt, #456,
                        model_name='kEffNet-b5',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        activation_fn=activation_fn,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)


def kEfficientNetB6(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling='avg',
                   classes=1000,
                   kType=2,
                   dropout_rate=0.5,
                   drop_connect_rate=0.2,
                   skip_stride_cnt=-1, 
                   activation_fn=swish,
                   dropout_all_blocks=False,
                   **kwargs):
    return kEfficientNet(1.8, 2.6, skip_stride_cnt=skip_stride_cnt, #528,
                        model_name='kEffNet-b6',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        activation_fn=activation_fn,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)


def kEfficientNetB7(include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling='avg',
                   classes=1000,
                   kType=2,
                   dropout_rate=0.5,
                   drop_connect_rate=0.2,
                   skip_stride_cnt=-1, 
                   activation_fn=swish,
                   dropout_all_blocks=False,
                   **kwargs):
    return kEfficientNet(2.0, 3.1, skip_stride_cnt=skip_stride_cnt, #600,
                        model_name='kEffNet-b7',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        activation_fn=activation_fn,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)


def kEfficientNetBN(N=0, 
                   include_top=True,
                   input_tensor=None,
                   input_shape=None,
                   pooling='avg',
                   classes=1000,
                   kType=2,
                   dropout_rate=None,
                   drop_connect_rate=0.2,
                   skip_stride_cnt=-1,
                   dropout_all_blocks=False,
                   **kwargs):
    result = None
    if (N==-1):
        dropout_rate=0.2 if dropout_rate is None else dropout_rate
        result = kEfficientNet(1.0, 1.0, skip_stride_cnt=skip_stride_cnt, # 224,
                            model_name='kEffNet-s',
                            include_top=include_top,
                            input_tensor=input_tensor, input_shape=input_shape,
                            pooling=pooling, classes=classes,
                            kType=kType,
                            dropout_rate=dropout_rate,
                            drop_connect_rate=drop_connect_rate,
                            blocks_args=SHALLOW_BLOCKS_ARGS,
                            dropout_all_blocks=dropout_all_blocks,
                            **kwargs)
    if (N==0):
        dropout_rate=0.2 if dropout_rate is None else dropout_rate
        result = kEfficientNet(1.0, 1.0, skip_stride_cnt=skip_stride_cnt, # 224,
                            model_name='kEffNet-b0',
                            include_top=include_top,
                            input_tensor=input_tensor, input_shape=input_shape,
                            pooling=pooling, classes=classes,
                            kType=kType,
                            dropout_rate=dropout_rate,
                            drop_connect_rate=drop_connect_rate,
                            dropout_all_blocks=dropout_all_blocks,
                            **kwargs)
    elif (N==1):
        dropout_rate=0.2 if dropout_rate is None else dropout_rate
        result = kEfficientNet(1.0, 1.1, skip_stride_cnt=skip_stride_cnt, #240,
                        model_name='kEffNet-b1',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)
    elif (N==2):
        dropout_rate=0.3 if dropout_rate is None else dropout_rate
        result = kEfficientNet(1.1, 1.2, skip_stride_cnt=skip_stride_cnt, #260,
                        model_name='kEffNet-b2',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)
    elif (N==3):
        dropout_rate=0.3 if dropout_rate is None else dropout_rate
        result = kEfficientNet(1.2, 1.4, skip_stride_cnt=skip_stride_cnt, #300,
                        model_name='kEffNet-b3',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)
    elif (N==4):
        dropout_rate=0.4 if dropout_rate is None else dropout_rate
        result = kEfficientNet(1.4, 1.8, skip_stride_cnt=skip_stride_cnt, #380,
                        model_name='kEffNet-b4',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)
    elif (N==5):
        dropout_rate=0.4 if dropout_rate is None else dropout_rate
        result = kEfficientNet(1.6, 2.2, skip_stride_cnt=skip_stride_cnt, #456,
                        model_name='kEffNet-b5',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)
    elif (N==6):
        dropout_rate=0.5 if dropout_rate is None else dropout_rate
        result = kEfficientNet(1.8, 2.6, skip_stride_cnt=skip_stride_cnt, #528,
                        model_name='kEffNet-b6',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)
    elif (N==7):
        dropout_rate=0.5 if dropout_rate is None else dropout_rate
        result = kEfficientNet(2.0, 3.1, skip_stride_cnt=skip_stride_cnt, #600,
                        model_name='kEffNet-b7',
                        include_top=include_top,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        kType=kType,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        dropout_all_blocks=dropout_all_blocks,
                        **kwargs)
    return result

def preprocess_input(x, data_format=None, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format,
                                           mode='torch', **kwargs)


setattr(EfficientNetB0, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB1, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB2, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB3, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB4, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB5, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB6, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB7, '__doc__', EfficientNet.__doc__)
