import keras
from keras import backend
import cai.datasets
import cai.layers
from keras.models import Model
from keras.datasets import cifar10

def lrscheduler(epoch):
  if epoch < 150:
    return 0.1
  elif epoch < 225:
    return 0.01
  else:
    return 0.001

def densenet_conv_block(x, growth_rate, bottleneck, l2_decay, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        bottleneck: float, densenet bottleneck.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    if bottleneck > 0:
        x1 = keras.layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,                                   
                                   name=name + '_0_bn')(x)
        x1 = keras.layers.Activation('relu', name=name + '_0_relu')(x1)
        x1 = keras.layers.Conv2D(bottleneck, 1,
                       use_bias=False,
                       kernel_regularizer=keras.regularizers.l2(l2_decay),
                       name=name + '_1_conv')(x1)
        x1 = keras.layers.BatchNormalization(axis=bn_axis, 
                                   epsilon=1.001e-5,                                   
                                   name=name + '_1_bn')(x1)
    else:
        x1 = keras.layers.BatchNormalization(axis=bn_axis, 
                                   epsilon=1.001e-5,                                   
                                   name=name + '_1_bn')(x)
    x1 = keras.layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = keras.layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       kernel_regularizer=keras.regularizers.l2(l2_decay),
                       name=name + '_2_conv')(x1)
    x = keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def densenet_block(x, blocks, growth_rate, bottleneck, l2_decay, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        growth_rate: float, growth rate at dense layers.
        bottleneck: float, densenet bottleneck.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = densenet_conv_block(x, growth_rate, bottleneck, l2_decay, name=name + '_block' + str(i + 1))
    return x
    
def densenet_transition_block(x, reduction, l2_decay, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_relu')(x)
    if reduction < 1: 
       x = keras.layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2_decay),
                      name=name + '_conv')(x)
    x = keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x
    
def simple_densenet(pinput_shape, blocks=6, growth_rate=12, bottleneck=48, compression=0.5,  l2_decay=0.000001,  num_classes=10):
    bn_axis = 3
    img_input = keras.layers.Input(shape=pinput_shape)
    x = keras.layers.Conv2D(24, (3, 3), padding='same',
                     input_shape=pinput_shape, 
                     kernel_regularizer=keras.regularizers.l2(l2_decay))(img_input)
    x = densenet_block(x, blocks, growth_rate, bottleneck, l2_decay, name='dn1')
    x = densenet_transition_block(x, compression, l2_decay, name='dntransition1')
    x = densenet_block(x, blocks, growth_rate, bottleneck, l2_decay, name='dn2')
    x = densenet_transition_block(x, compression, l2_decay, name='dntransition2')
    x = densenet_block(x, blocks, growth_rate, bottleneck, l2_decay, name='dn3')
    x = keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = keras.layers.Activation('relu', name='relu')(x)
    x = keras.layers.GlobalAveragePooling2D(name='last_avg_pool')(x)
    x = keras.layers.Dense(num_classes, activation='softmax', name='softmax')(x)            
    return Model(inputs = [img_input], outputs = [x])
    
def two_paths_densenet(pinput_shape, blocks=6, growth_rate=12, bottleneck=48, compression=0.5,  l2_decay=0.000001, num_classes=10):
    bn_axis = 3
    img_input = keras.layers.Input(shape=pinput_shape)
    x = cai.layers.CopyChannels(0,1)(img_input)
    x = keras.layers.Conv2D(16, (3, 3), padding='same',
                     input_shape=pinput_shape, 
                     kernel_regularizer=keras.regularizers.l2(l2_decay))(x)
    x = densenet_block(x, int(blocks/2), growth_rate, bottleneck, l2_decay, name='L1')
    x = densenet_transition_block(x, compression, l2_decay, name='LTRANS')

    x2 = cai.layers.CopyChannels(1,2)(img_input)
    x2 = keras.layers.Conv2D(8, (3, 3), padding='same',
                     input_shape=pinput_shape, 
                     kernel_regularizer=keras.regularizers.l2(l2_decay))(x2)
    x2 = densenet_block(x2, int(blocks/2), growth_rate, bottleneck, l2_decay, name='AB1')
    x2 = densenet_transition_block(x2, compression, l2_decay, name='ABTRANS')
    x = keras.layers.Concatenate(axis=bn_axis, name='concat')([x, x2])
    
    x = densenet_block(x, blocks, growth_rate, bottleneck, l2_decay, name='dn2')
    x = densenet_transition_block(x, compression, l2_decay, name='dntransition2')
    x = densenet_block(x, blocks, growth_rate, bottleneck, l2_decay, name='dn3')
    x = keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = keras.layers.Activation('relu', name='relu')(x)
    x = keras.layers.GlobalAveragePooling2D(name='last_avg_pool')(x)
    x = keras.layers.Dense(num_classes, activation='softmax', name='softmax')(x)            
    return Model(inputs = [img_input], outputs = [x])

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
