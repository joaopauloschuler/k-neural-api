"""Functions to create DenseNet architectures.
https://arxiv.org/abs/1608.06993
"""

import keras
from keras import backend
import cai.datasets
import cai.layers
from keras.models import Model
from keras.datasets import cifar10

def lrscheduler(epoch):
    """Default DenseNet Learning Rate Scheduler.
    # Arguments
        epoch: integer with current epoch count.
    # Returns
        float with desired learning rate.
    """
    if epoch < 150:
       return 0.1
    elif epoch < 225:
       return 0.01
    else:
       return 0.001

def densenet_conv_block(last_tensor, growth_rate, bottleneck, l2_decay, name):
    """Builds a unit inside a densenet convolutional block.
    # Arguments
        last_tensor: input tensor.
        growth_rate: float, growth rate at dense layers.
        bottleneck: float, densenet bottleneck.
        l2_decay: float.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    if bottleneck > 0:
        x1 = keras.layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5)(last_tensor)
        x1 = keras.layers.Activation('relu')(x1)
        x1 = keras.layers.Conv2D(bottleneck, 1,
                       use_bias=False,
                       kernel_regularizer=keras.regularizers.l2(l2_decay))(x1)
        x1 = keras.layers.BatchNormalization(axis=bn_axis, 
                                   epsilon=1.001e-5)(x1)
    else:
        x1 = keras.layers.BatchNormalization(axis=bn_axis, 
                                   epsilon=1.001e-5)(last_tensor)
    x1 = keras.layers.Activation('relu')(x1)
    x1 = keras.layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       kernel_regularizer=keras.regularizers.l2(l2_decay))(x1)
    last_tensor = keras.layers.Concatenate(axis=bn_axis)([last_tensor, x1])
    return last_tensor

def densenet_block(last_tensor, blocks, growth_rate, bottleneck, l2_decay, name):
    """Builds a densenet convolutional block.
    # Arguments
        last_tensor: input tensor.
        blocks: integer, the number of building blocks.
        growth_rate: float, growth rate at dense layers.
        bottleneck: float, densenet bottleneck.
        l2_decay: float.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        last_tensor = densenet_conv_block(last_tensor, growth_rate, bottleneck, l2_decay, name=name + '_b' + str(i + 1))
    return last_tensor
    
def densenet_transition_block(last_tensor, compression, l2_decay, name):
    """Builds a densenet transition block.
    # Arguments
        last_tensor: input tensor.
        compression: float, compression rate at transition layers.
        l2_decay: float.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    last_tensor = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(last_tensor)
    last_tensor = keras.layers.Activation('relu')(last_tensor)
    last_tensor = keras.layers.Conv2D(int(backend.int_shape(last_tensor)[bn_axis] * compression), 1,
                      use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2_decay))(last_tensor)
    last_tensor = keras.layers.AveragePooling2D(2, strides=2)(last_tensor)
    return last_tensor
    
def simple_densenet(pinput_shape, blocks=6, growth_rate=12, bottleneck=48, compression=0.5,  l2_decay=0.000001,  num_classes=10):
    """Builds a simple densenet model from input to end.
    # Arguments
        pinput_shape: array with input shape.
        blocks: integer number with densenet number of blocks.
        growth_rate: integer number with the number of channels added at each convolution.
        bottleneck: integer. This is the number of bottleneck output channels.
        compression: compression rate at transition blocks.
        l2_decay: float.
        num_classes: integer number with the number of classes to be classified.
    # Returns
        a densenet model.
    """
    bn_axis = 3
    img_input = keras.layers.Input(shape=pinput_shape)
    last_tensor = keras.layers.Conv2D(24, (3, 3), padding='same',
                     input_shape=pinput_shape, 
                     kernel_regularizer=keras.regularizers.l2(l2_decay))(img_input)
    last_tensor = densenet_block(last_tensor, blocks, growth_rate, bottleneck, l2_decay, name='dn1')
    last_tensor = densenet_transition_block(last_tensor, compression, l2_decay, name='dntransition1')
    last_tensor = densenet_block(last_tensor, blocks, growth_rate, bottleneck, l2_decay, name='dn2')
    last_tensor = densenet_transition_block(last_tensor, compression, l2_decay, name='dntransition2')
    last_tensor = densenet_block(last_tensor, blocks, growth_rate, bottleneck, l2_decay, name='dn3')
    last_tensor = keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(last_tensor)
    last_tensor = keras.layers.Activation('relu', name='relu')(last_tensor)
    #Makes sense testing this:
    #last_tensor = keras.layers.Conv2D(int(backend.int_shape(last_tensor)[bn_axis] * compression), 1,
    #                  use_bias=False,
    #                  kernel_regularizer=keras.regularizers.l2(l2_decay))(last_tensor)
    last_tensor = keras.layers.GlobalAveragePooling2D(name='last_avg_pool')(last_tensor)
    last_tensor = keras.layers.Dense(num_classes, activation='softmax', name='softmax')(last_tensor)            
    return Model(inputs = [img_input], outputs = [last_tensor])
    
def two_paths_densenet(pinput_shape, blocks=6, growth_rate=12, bottleneck=48, compression=0.5,  l2_decay=0.000001, num_classes=10):
    """Builds a two-paths optimized densenet model from input to end.
    # Arguments
        pinput_shape: array with input shape.
        blocks: integer number with densenet number of blocks.
        growth_rate: integer number with the number of channels added at each convolution.
        bottleneck: integer. This is the number of bottleneck output channels.
        compression: compression rate at transition blocks.
        l2_decay: float.
        num_classes: integer number with the number of classes to be classified.
    # Returns
        a two paths densenet model.
    """
    bn_axis = 3
    img_input = keras.layers.Input(shape=pinput_shape)
    last_tensor = cai.layers.CopyChannels(0,1)(img_input)
    last_tensor = keras.layers.Conv2D(16, (3, 3), padding='same',
                     input_shape=pinput_shape, 
                     kernel_regularizer=keras.regularizers.l2(l2_decay))(last_tensor)
    last_tensor = densenet_block(last_tensor, blocks, int(growth_rate/2), int(bottleneck/2), l2_decay, name='L')
    last_tensor = densenet_transition_block(last_tensor, compression, l2_decay, name='LTRANS')

    x2 = cai.layers.CopyChannels(1,2)(img_input)
    x2 = keras.layers.Conv2D(8, (3, 3), padding='same',
                     input_shape=pinput_shape, 
                     kernel_regularizer=keras.regularizers.l2(l2_decay))(x2)
    x2 = densenet_block(x2, blocks, int(growth_rate/2), int(bottleneck/2), l2_decay, name='AB')
    x2 = densenet_transition_block(x2, compression, l2_decay, name='ABTRANS')
    last_tensor = keras.layers.Concatenate(axis=bn_axis, name='concat')([last_tensor, x2])
    
    last_tensor = densenet_block(last_tensor, blocks, growth_rate, bottleneck, l2_decay, name='dn2')
    last_tensor = densenet_transition_block(last_tensor, compression, l2_decay, name='dntransition2')
    last_tensor = densenet_block(last_tensor, blocks, growth_rate, bottleneck, l2_decay, name='dn3')
    last_tensor = keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(last_tensor)
    last_tensor = keras.layers.Activation('relu', name='relu')(last_tensor)
    #Makes sense testing this:
    #last_tensor = keras.layers.Conv2D(int(backend.int_shape(last_tensor)[bn_axis] * compression), 1,
    #                  use_bias=False,
    #                  kernel_regularizer=keras.regularizers.l2(l2_decay))(last_tensor)
    last_tensor = keras.layers.GlobalAveragePooling2D(name='last_avg_pool')(last_tensor)
    last_tensor = keras.layers.Dense(num_classes, activation='softmax', name='softmax')(last_tensor)            
    return Model(inputs = [img_input], outputs = [last_tensor])

def train_simple_densenet_on_dataset(base_model_name, dataset, input_shape, blocks=6, growth_rate=12, bottleneck=48, compression=0.5,  l2_decay=0.00001,  batch_size=64, epochs=300):
    batches_per_epoch = int(40000/batch_size)
    l2_decay = l2_decay / batches_per_epoch
    model = simple_densenet(input_shape, blocks=blocks, growth_rate=growth_rate, bottleneck=bottleneck, compression=compression,  l2_decay=l2_decay)
    fit_result,  model_name,  csv_name = cai.datasets.train_model_on_cifar10(model,  base_model_name, plrscheduler=lrscheduler,  batch_size=batch_size, epochs=epochs)    
    return model, fit_result,  model_name,  csv_name
    
def train_simple_densenet_on_cifar10(base_model_name, blocks=6, growth_rate=12, bottleneck=48, compression=0.5,  l2_decay=0.00001,  batch_size=64, epochs=300):
    input_shape=[32, 32, 3]
    return train_simple_densenet_on_dataset(base_model_name=base_model_name, dataset=cifar10, input_shape=input_shape, 
      blocks=blocks, growth_rate=growth_rate, bottleneck=bottleneck, compression=compression,  l2_decay=l2_decay, 
      batch_size=batch_size, epochs=epochs)
