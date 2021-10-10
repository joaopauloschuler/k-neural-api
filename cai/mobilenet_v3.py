"""MobileNet v3 models for Keras.

The following table describes the performance of MobileNets:
------------------------------------------------------------------------
MACs stands for Multiply Adds

| Classification Checkpoint| MACs(M)| Parameters(M)| Top1 Accuracy| Pixel1 CPU(ms)|

| [mobilenet_v3_large_1.0_224]              | 217 | 5.4 |   75.6   |   51.2   |
| [mobilenet_v3_large_0.75_224]             | 155 | 4.0 |   73.3   |   39.8   |
| [mobilenet_v3_large_minimalistic_1.0_224] | 209 | 3.9 |   72.3   |   44.1   |
| [mobilenet_v3_small_1.0_224]              | 66  | 2.9 |   68.1   |   15.8   |
| [mobilenet_v3_small_0.75_224]             | 44  | 2.4 |   65.4   |   12.8   |
| [mobilenet_v3_small_minimalistic_1.0_224] | 65  | 2.0 |   61.9   |   12.2   |

The weights for all 6 models are obtained and
translated from the Tensorflow checkpoints
from TensorFlow checkpoints found [here]
(https://github.com/tensorflow/models/tree/master/research/
slim/nets/mobilenet/README.md).

# Reference

This file contains building code for MobileNetV3, based on
[Searching for MobileNetV3]
(https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)

COPYRIGHT

Copyright (c) 2016 - 2018, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.
The initial code of this file came from https://github.com/keras-team/keras-applications/edit/master/keras_applications/mobilenet_v3.py
(the Keras repository), hence, for author information regarding commits
that occured earlier than the first commit in the present repository,
please see the original Keras repository.

The original file from above link was modified. Modifications can be tracked via 
git commits at https://github.com/joaopauloschuler/k-neural-api/blob/master/cai/mobilenet_v3.py

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

import keras.utils
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend

import cai.layers

BASE_WEIGHT_PATH = ('https://github.com/DrSlink/mobilenet_v3_keras/'
                    'releases/download/v1.0/')
WEIGHTS_HASHES = {
    'large_224_0.75_float': (
        '765b44a33ad4005b3ac83185abf1d0eb',
        'c256439950195a46c97ede7c294261c6'),
    'large_224_1.0_float': (
        '59e551e166be033d707958cf9e29a6a7',
        '12c0a8442d84beebe8552addf0dcb950'),
    'large_minimalistic_224_1.0_float': (
        '675e7b876c45c57e9e63e6d90a36599c',
        'c1cddbcde6e26b60bdce8e6e2c7cae54'),
    'small_224_0.75_float': (
        'cb65d4e5be93758266aa0a7f2c6708b7',
        'c944bb457ad52d1594392200b48b4ddb'),
    'small_224_1.0_float': (
        '8768d4c2e7dee89b9d02b2d03d65d862',
        '5bec671f47565ab30e540c257bba8591'),
    'small_minimalistic_224_1.0_float': (
        '99cd97fb2fcdad2bf028eb838de69e37',
        '1efbf7e822e03f250f45faa3c6bbe156'),
}

def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
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

# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/
# slim/nets/mobilenet/mobilenet.py


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(inputs)
    if backend.image_data_format() == 'channels_first':
        x = layers.Reshape((filters, 1, 1))(x)
    else:
        x = layers.Reshape((1, 1, filters))(x)
    x = layers.Conv2D(_depth(filters * se_ratio),
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'squeeze_excite/Conv')(x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(filters,
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'squeeze_excite/Conv_1')(x)
    x = layers.Activation(cai.layers.HardSigmoid)(x)
    if backend.backend() == 'theano':
        # For the Theano backend, we have to explicitly make
        # the excitation weights broadcastable.
        x = layers.Lambda(
            lambda br: backend.pattern_broadcast(br, [True, True, True, False]),
            output_shape=lambda input_shape: input_shape,
            name=prefix + 'squeeze_excite/broadcast')(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride,
                        se_ratio, activation, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(_depth(infilters * expansion),
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand/BatchNorm')(x)
        x = layers.Activation(activation)(x)

    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, kernel_size),
                                 name=prefix + 'depthwise/pad')(x)
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=stride,
                               padding='same' if stride == 1 else 'valid',
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise/BatchNorm')(x)
    x = layers.Activation(activation)(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project/BatchNorm')(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x


def MobileNetV3(stack_fn,
                last_point_ch,
                input_shape=None,
                alpha=1.0,
                model_type='large',
                minimalistic=False,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                classes=1000,
                pooling=None,
                dropout_rate=0.2,
                **kwargs):
    """Instantiates the MobileNetV3 architecture.

    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        last_point_ch: number channels at the last layer (before top)
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
            depth multiplier in the MobileNetV3 paper, but the name is kept for
            consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                are used at each layer.
        model_type: MobileNetV3 is defined as two models: large and small. These
        models are targeted at high and low resource use cases respectively.
        minimalistic: In addition to large and small models this module also contains
            so-called minimalistic models, these models have the same per-layer
            dimensions characteristic as MobilenetV3 however, they don't utilize any
            of the advanced blocks (squeeze-and-excite units, hard-swish, and 5x5
            convolutions). While these models are less efficient on CPU, they are
            much more performant on GPU/DSP.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
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
        dropout_rate: fraction of the input units to drop on the last layer
    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid model type, argument for `weights`,
            or invalid input shape when weights='imagenet'
    """
    
    img_input = keras.layers.Input(shape=input_shape)
    
    channel_axis = cai.layers.GetChannelAxis()
    
    if minimalistic:
        kernel = 3
        activation = 'relu'
        se_ratio = None
    else:
        kernel = 5
        activation = cai.layers.HardSwish
        se_ratio = 0.25

    x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
                             name='Conv_pad')(img_input)
    x = layers.Conv2D(16,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv/BatchNorm')(x)
    x = layers.Activation(activation)(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)

    x = layers.Conv2D(last_conv_ch,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name='Conv_1')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv_1/BatchNorm')(x)
    x = layers.Activation(activation)(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        if channel_axis == 1:
            x = layers.Reshape((last_conv_ch, 1, 1))(x)
        else:
            x = layers.Reshape((1, 1, last_conv_ch))(x)
        x = layers.Conv2D(last_point_ch,
                          kernel_size=1,
                          padding='same',
                          name='Conv_2')(x)
        x = layers.Activation(activation)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(classes,
                          kernel_size=1,
                          padding='same',
                          name='Logits')(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name='Predictions/Softmax')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    inputs = img_input
    
    # Create model.
    model = keras.models.Model(inputs, x, name='MobilenetV3' + model_type)

    return model


def MobileNetV3Small(input_shape=None,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     classes=1000,
                     pooling=None,
                     dropout_rate=0.2,
                     **kwargs):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)
        x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, 'relu', 0)
        x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, 'relu', 1)
        x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, 'relu', 2)
        x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
        x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10)
        return x
    return MobileNetV3(stack_fn,
                       1024,
                       input_shape,
                       alpha,
                       'small',
                       minimalistic,
                       include_top,
                       weights,
                       input_tensor,
                       classes,
                       pooling,
                       dropout_rate,
                       **kwargs)


def MobileNetV3Large(input_shape=None,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     classes=1000,
                     pooling=None,
                     dropout_rate=0.2,
                     **kwargs):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)
        x = _inverted_res_block(x, 1, depth(16), 3, 1, None, 'relu', 0)
        x = _inverted_res_block(x, 4, depth(24), 3, 2, None, 'relu', 1)
        x = _inverted_res_block(x, 3, depth(24), 3, 1, None, 'relu', 2)
        x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, 'relu', 3)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, 'relu', 4)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, 'relu', 5)
        x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
        x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
        x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio,
                                activation, 12)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio,
                                activation, 13)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio,
                                activation, 14)
        return x
    return MobileNetV3(stack_fn,
                       1280,
                       input_shape,
                       alpha,
                       'large',
                       minimalistic,
                       include_top,
                       weights,
                       input_tensor,
                       classes,
                       pooling,
                       dropout_rate,
                       **kwargs)

def kse_block(inputs, filters, se_ratio, prefix, kType=0):
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(inputs)
    if backend.image_data_format() == 'channels_first':
        x = layers.Reshape((filters, 1, 1))(x)
        channel_axis = 1
    else:
        x = layers.Reshape((1, 1, filters))(x)
        channel_axis = 3
    #x = layers.Conv2D(_depth(filters * se_ratio),
    #                  kernel_size=1,
    #                  padding='same',
    #                  name=prefix + 'squeeze_excite/Conv')(x)
    # x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = cai.layers.kPointwiseConv2D(x, filters=_depth(filters * se_ratio), channel_axis=channel_axis, name=prefix + 'squeeze_excite/Conv', activation='relu', has_batch_norm=False, use_bias=True, kType=kType)
    # x = layers.Conv2D(filters,
    #                  kernel_size=1,
    #                  padding='same',
    #                  name=prefix + 'squeeze_excite/Conv_1')(x)
    #x = layers.Activation(hard_sigmoid)(x)
    x = cai.layers.kPointwiseConv2D(x, filters=filters, channel_axis=channel_axis, name=prefix + 'squeeze_excite/Conv_1', activation=cai.layers.HardSigmoid, has_batch_norm=False, use_bias=True, kType=kType)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x

def kinverted_res_block(x, expansion, filters, kernel_size, stride,
                        se_ratio, activation, block_id,  kType=0):
    channel_axis = cai.layers.GetChannelAxis()
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        # x = layers.Conv2D(_depth(infilters * expansion),
        #                   kernel_size=1,
        #                  padding='same',
        #                  use_bias=False,
        #                  name=prefix + 'expand')(x)
        # x = layers.BatchNormalization(axis=channel_axis,
        #                              epsilon=1e-3,
        #                              momentum=0.999,
        #                              name=prefix + 'expand/BatchNorm')(x)
        #x = layers.Activation(activation)(x)
        x = cai.layers.kPointwiseConv2D(x, filters=_depth(infilters * expansion), channel_axis=channel_axis, name=prefix + 'expand', activation=activation, has_batch_norm=True, use_bias=False, kType=kType)

    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, kernel_size),
                                 name=prefix + 'depthwise/pad')(x)
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=stride,
                               padding='same' if stride == 1 else 'valid',
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise/BatchNorm')(x)
    x = layers.Activation(activation)(x)

    if se_ratio:
        x = kse_block(x, _depth(infilters * expansion), se_ratio, prefix, kType=kType)

    # x = layers.Conv2D(filters,
    #                  kernel_size=1,
    #                  padding='same',
    #                  use_bias=False,
    #                  name=prefix + 'project')(x)
    # x = layers.BatchNormalization(axis=channel_axis,
    #                              epsilon=1e-3,
    #                              momentum=0.999,
    #                              name=prefix + 'project/BatchNorm')(x)
    x = cai.layers.kPointwiseConv2D(x, filters=filters, channel_axis=channel_axis, name=prefix + 'project', activation=None, has_batch_norm=True, use_bias=False, kType=kType)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x

def kMobileNetV3(stack_fn,
                last_point_ch,
                input_shape=None,
                alpha=1.0,
                model_type='large',
                minimalistic=False,
                include_top=True,
                input_tensor=None,
                classes=1000,
                pooling=None,
                dropout_rate=0.2,
                kType=0,
                **kwargs):
    """Instantiates the MobileNetV3 architecture.

    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        last_point_ch: number channels at the last layer (before top)
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
            depth multiplier in the MobileNetV3 paper, but the name is kept for
            consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                are used at each layer.
        model_type: MobileNetV3 is defined as two models: large and small. These
        models are targeted at high and low resource use cases respectively.
        minimalistic: In addition to large and small models this module also contains
            so-called minimalistic models, these models have the same per-layer
            dimensions characteristic as MobilenetV3 however, they don't utilize any
            of the advanced blocks (squeeze-and-excite units, hard-swish, and 5x5
            convolutions). While these models are less efficient on CPU, they are
            much more performant on GPU/DSP.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
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
        dropout_rate: fraction of the input units to drop on the last layer
    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid model type, argument for `weights`,
            or invalid input shape when weights='imagenet'
    """
    
    img_input = keras.layers.Input(shape=input_shape)
    
    channel_axis = cai.layers.GetChannelAxis()

    if minimalistic:
        kernel = 3
        activation = 'relu'
        se_ratio = None
    else:
        kernel = 5
        activation = cai.layers.HardSwish
        se_ratio = 0.25

    x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
                             name='Conv_pad')(img_input)
    x = layers.Conv2D(16,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv/BatchNorm')(x)
    x = layers.Activation(activation)(x)

    x = stack_fn(x, kernel, activation, se_ratio, kType=kType)

    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)

    # x = layers.Conv2D(last_conv_ch,
    #                  kernel_size=1,
    #                  padding='same',
    #                  use_bias=False,
    #                  name='Conv_1')(x)
    # x = layers.BatchNormalization(axis=channel_axis,
    #                              epsilon=1e-3,
    #                              momentum=0.999,
    #                              name='Conv_1/BatchNorm')(x)
    # x = layers.Activation(activation)(x)
    x = cai.layers.kPointwiseConv2D(x, filters=last_conv_ch, channel_axis=channel_axis, name='Conv_1', activation=activation, has_batch_norm=True, use_bias=False, kType=kType)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        if channel_axis == 1:
            x = layers.Reshape((last_conv_ch, 1, 1))(x)
        else:
            x = layers.Reshape((1, 1, last_conv_ch))(x)
        #x = layers.Conv2D(last_point_ch,
        #                  kernel_size=1,
        #                  padding='same',
        #                  name='Conv_2')(x)
        #x = layers.Activation(activation)(x)
        x = cai.layers.kPointwiseConv2D(x, filters=last_point_ch, channel_axis=channel_axis, name='Conv_2', activation=activation, has_batch_norm=False, use_bias=True, kType=kType)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        #Last layer hasn't been transformed.
        x = layers.Conv2D(classes,
                          kernel_size=1,
                          padding='same',
                          name='Logits')(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name='Predictions/Softmax')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    inputs = img_input
    
    # Create model.
    model = keras.models.Model(inputs, x, name='kMobilenetV3' + model_type+'-'+str(kType))

    return model

def kMobileNetV3Small(input_shape=None,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     input_tensor=None,
                     classes=1000,
                     pooling=None,
                     dropout_rate=0.2,
                     kType=0,
                     **kwargs):
    def stack_fn(x, kernel, activation, se_ratio, kType=0):
        def depth(d):
            return _depth(d * alpha)
        x = kinverted_res_block(x, 1, depth(16), 3, 2, se_ratio, 'relu', 0, kType=kType)
        x = kinverted_res_block(x, 72. / 16, depth(24), 3, 2, None, 'relu', 1, kType=kType)
        x = kinverted_res_block(x, 88. / 24, depth(24), 3, 1, None, 'relu', 2, kType=kType)
        x = kinverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3, kType=kType)
        x = kinverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4, kType=kType)
        x = kinverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5, kType=kType)
        x = kinverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6, kType=kType)
        x = kinverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7, kType=kType)
        x = kinverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8, kType=kType)
        x = kinverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9, kType=kType)
        x = kinverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10, kType=kType)
        return x
    return kMobileNetV3(stack_fn,
                       1024,
                       input_shape,
                       alpha,
                       'small',
                       minimalistic,
                       include_top,
                       input_tensor,
                       classes,
                       pooling,
                       dropout_rate,
                       kType=kType,  
                       **kwargs)


def kMobileNetV3Large(input_shape=None,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     input_tensor=None,
                     classes=1000,
                     pooling=None,
                     dropout_rate=0.2,
                     kType=0,
                     **kwargs):
    def stack_fn(x, kernel, activation, se_ratio, kType=0):
        def depth(d):
            return _depth(d * alpha)
        x = kinverted_res_block(x, 1, depth(16), 3, 1, None, 'relu', 0, kType=kType)
        x = kinverted_res_block(x, 4, depth(24), 3, 2, None, 'relu', 1, kType=kType)
        x = kinverted_res_block(x, 3, depth(24), 3, 1, None, 'relu', 2, kType=kType)
        x = kinverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, 'relu', 3, kType=kType)
        x = kinverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, 'relu', 4, kType=kType)
        x = kinverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, 'relu', 5, kType=kType)
        x = kinverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6, kType=kType)
        x = kinverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7, kType=kType)
        x = kinverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8, kType=kType)
        x = kinverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9, kType=kType)
        x = kinverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10, kType=kType)
        x = kinverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11, kType=kType)
        x = kinverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12, kType=kType)
        x = kinverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13, kType=kType)
        x = kinverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14, kType=kType)
        return x
    return kMobileNetV3(stack_fn,
                       1280,
                       input_shape,
                       alpha,
                       'large',
                       minimalistic,
                       include_top,
                       input_tensor,
                       classes,
                       pooling,
                       dropout_rate,
                       kType=kType,
                       **kwargs)

setattr(MobileNetV3Small, '__doc__', MobileNetV3.__doc__)
setattr(MobileNetV3Large, '__doc__', MobileNetV3.__doc__)
