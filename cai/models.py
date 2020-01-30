import keras
from keras.models import Model
#from keras import backend
#import cai.datasets
#import cai.layers

def plant_leaf(pinput_shape, num_classes,  l2_decay=0.0, dropout_drop_rate=0.2, has_batch_norm=True):
    img_input = keras.layers.Input(shape=pinput_shape)
    last_tensor = keras.layers.Conv2D(32, (3, 3), padding='valid',
        input_shape=pinput_shape, 
        kernel_regularizer=keras.regularizers.l2(l2_decay))(img_input)
    if (has_batch_norm): last_tensor = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(last_tensor)
    last_tensor = keras.layers.Activation('relu')(last_tensor)
    last_tensor = keras.layers.MaxPooling2D(2, strides=2)(last_tensor)
    
    last_tensor = keras.layers.Conv2D(16, (3, 3), padding='valid',
        input_shape=pinput_shape, 
        kernel_regularizer=keras.regularizers.l2(l2_decay))(last_tensor)
    if (has_batch_norm): last_tensor = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(last_tensor)
    last_tensor = keras.layers.Activation('relu')(last_tensor)
    last_tensor = keras.layers.MaxPooling2D(2, strides=2)(last_tensor)
    
    last_tensor = keras.layers.Conv2D(16, (3, 3), padding='valid',
        input_shape=pinput_shape, 
        kernel_regularizer=keras.regularizers.l2(l2_decay))(last_tensor)
    if (has_batch_norm): last_tensor = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(last_tensor)
    last_tensor = keras.layers.Activation('relu')(last_tensor)
    last_tensor = keras.layers.MaxPooling2D(2, strides=2)(last_tensor)
    
    last_tensor = keras.layers.Flatten()(last_tensor)
    
    last_tensor = keras.layers.Dense(128)(last_tensor)
    if (dropout_drop_rate > 0.0):
        last_tensor = keras.layers.Dropout(rate=dropout_drop_rate)(last_tensor)
    last_tensor = keras.layers.Activation('relu')(last_tensor)
    
    last_tensor = keras.layers.Dense(num_classes, activation='softmax', name='softmax')(last_tensor)            
    return Model(inputs = [img_input], outputs = [last_tensor])
