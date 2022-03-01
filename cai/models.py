import cai.layers
import cai.util
import cai.inception_v3
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
import numpy as np

def save_model(model, base_name):
    """Saves a model with its weights.
    # Arguments
        model: Keras model instance to be saved.
        base_name: base file name
    """
    text_file = open(base_name+'.model', "w")
    text_file.write(model.to_json())
    text_file.close()
    model.save_weights(base_name+'.h5')

def load_model(base_name):
    """Loads a model with its weights
    # Arguments
        base_name: base file name
    # Returns
        a model with its weights.
    """
    model = Model()
    text_file = open(base_name+'.model',"r")
    config = text_file.read()
    model = model_from_json(config,  {'CopyChannels': cai.layers.CopyChannels})
    model.load_weights(base_name+'.h5')
    return model

def load_kereas_model(filename):
    """
    Loads a Keras model with CAI custom layers.
    """
    return keras.models.load_model(filename, custom_objects=cai.layers.GetClasses())

def make_model_trainable(model):
    """
    Makes all layers trainable.
    """
    for layer in model.layers:
        layer.trainable = True

def make_model_untrainable(model):
    """
    Makes all layers untrainable.
    """
    for layer in model.layers:
        layer.trainable = False

def plant_leaf(pinput_shape, num_classes,  l2_decay=0.0, dropout_drop_rate=0.2, has_batch_norm=True):
    """Implements the architecture found on the paper: 
        Identification of plant leaf diseases using a nine-layer deep convolutional neural network.
        https://www.sciencedirect.com/science/article/pii/S0045790619300023
    """
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
    
    last_tensor = keras.layers.Conv2D(8, (3, 3), padding='valid',
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

def CreatePartialModel(pModel, pOutputLayerName, hasGlobalAvg=False):
  """Creates a partial model up to the layer name defined in pOutputLayerName.
  # Arguments
    pModel: original model.
    pOutputLayerName: last layer in the partial model.
    hasGlobalAvg: when True, adds a global average pooling at the end.
  """
  inputs = pModel.input
  outputs = pModel.get_layer(pOutputLayerName).output
  if (hasGlobalAvg):
    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
  return keras.Model(inputs=inputs, outputs=outputs)

def CreatePartialModelCopyingChannels(pModel, pOutputLayerName, pChannelStart, pChannelCount):
  """Creates a partial model up to the layer name defined in pOutputLayerName and then copies channels starting from pChannelStart with pChannelCount channels.
  # Arguments
    pModel: original model.
    pOutputLayerName: last layer in the partial model.
    hasGlobalAvg: when True, adds a global average pooling at the end.
    pChannelStart: first channel to be loaded.
    pChannelCount: channels to be loaded.    
  """
  inputs = pModel.input
  outputs = pModel.get_layer(pOutputLayerName).output
  outputs = cai.layers.CopyChannels(channel_start=pChannelStart, channel_count=pChannelCount)(outputs)
  return keras.Model(inputs=inputs, outputs=outputs)

def CreatePartialModelWithSoftMax(pModel, pOutputLayerName, numClasses, newLayerName='k_probs'):
  """Creates a partial model up to the layer name defined in pOutputLayerName and then adds a dense layer with softmax.
  This method was built to be used for image classification with transfer learning.
  # Arguments
    pModel: original model.
    pOutputLayerName: last layer in the partial model.
    numClasses: number of classes.
    newLayerName: last dense/softmax layer name.
  """
  model = cai.models.CreatePartialModel(pModel, pOutputLayerName)
  inputs = model.input
  outputs = model.get_layer(pOutputLayerName).output
  outputs = keras.layers.Dense(numClasses,
    activation='softmax',
    name=newLayerName)(outputs)
  model = keras.models.Model(inputs, outputs)
  return model

def CreatePartialModelFromChannel(pModel, pOutputLayerName, pChannelIdx):
  """Creates a partial model up to the layer name defined in pOutputLayerName and then copies the channel at index pChannelIdx.
  # Arguments
    pModel: original model.
    pOutputLayerName: last layer in the partial model.
    hasGlobalAvg: when True, adds a global average pooling at the end.
    pChannelIdx: output channel index to create the model from.
  """
  return CreatePartialModelCopyingChannels(pModel, pOutputLayerName, pChannelStart=pChannelIdx, pChannelCount=1)

def PartialModelPredict(aInput, pModel, pOutputLayerName, hasGlobalAvg=False, pBatchSize=32):
  """Creates a partial model up to the layer name defined in pOutputLayerName and run it
  with aInput.  
  # Arguments
    aInput: array with Input elements.
    pModel: original model.
    pOutputLayerName: last layer in the partial model.
    hasGlobalAvg: when True, adds a global average pooling at the end. 
  """  
  inputs = pModel.input
  outputs = pModel.get_layer(pOutputLayerName).output
  if (hasGlobalAvg):
    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
  IntermediateLayerModel = keras.Model(inputs=inputs, outputs=outputs)
  layeroutput = np.array(IntermediateLayerModel.predict(x=aInput, batch_size=pBatchSize))
  IntermediateLayerModel = 0
  return layeroutput

def calculate_heat_map_from_dense_and_avgpool(aInput, target_class, pModel, pOutputLayerName, pDenseLayerName):
  """Creates a heatmap from a model that has a global avg pool followed by a dense layer.  
  # Arguments
    aInput: array with Input elements.
    pModel: original model.
    pOutputLayerName: last layer before the global avg pool.
    pDenseLayerName: dense layer found probably before a softmax.
  """
  localImageArray = []
  localImageArray.append(aInput)
  localImageArray = np.array(localImageArray)
  class_weights = pModel.get_layer(pDenseLayerName).get_weights()[0]
  conv_output = cai.models.PartialModelPredict(localImageArray, pModel, pOutputLayerName)[0]
  a_heatmap_result = np.zeros(dtype = np.float32, shape = conv_output.shape[0:2])
  #print(a_heatmap_result.shape)
  #print(type(conv_output[:, :, 0]))
  #print(conv_output[:, :, 0].shape)
  for i, w in enumerate(class_weights[:, target_class]):
    a_heatmap_result += w * conv_output[:, :, i]
  a_heatmap_result = cai.util.relu(a_heatmap_result)
  max_heatmap_result = np.max(a_heatmap_result)
  if max_heatmap_result > 0:
      a_heatmap_result = a_heatmap_result / max_heatmap_result 
  return a_heatmap_result
  
  def compiled_full_two_path_inception_v3(
    input_shape=(224,224,3),
    classes=1000,
    max_mix_idx=10, 
    model_name='two_path_inception_v3'):
    """
    Wrapper for cai.inception_v3.compiled_two_path_inception_v3 .
    """
    return cai.inception_v3.compiled_two_path_inception_v3(
        input_shape=input_shape,
        classes=classes,
        two_paths_partial_first_block=0,
        two_paths_first_block=True,
        two_paths_second_block=True,
        deep_two_paths=True,
        deep_two_paths_compression=0.655,
        max_mix_idx=max_mix_idx, 
        model_name='deep_two_path_inception_v3'
    )
    
def compiled_inception_v3(
    input_shape=(224,224,3),
    classes=1000,
    max_mix_idx=10, 
    model_name='two_path_inception_v3'):
    """
    Wrapper for cai.inception_v3.compiled_two_path_inception_v3 .
    """
    return cai.inception_v3.compiled_two_path_inception_v3(
        input_shape=input_shape,
        classes=classes,
        two_paths_partial_first_block=0,
        two_paths_first_block=False,
        two_paths_second_block=False,
        deep_two_paths=False,
        max_mix_idx=max_mix_idx, 
        model_name='two_path_inception_v3'
    )

def compiled_two_path_inception_v3(
    input_shape=(224,224,3),
    classes=1000,
    two_paths_partial_first_block=0,
    two_paths_first_block=False,
    two_paths_second_block=False,
    deep_two_paths=False,
    deep_two_paths_compression=0.655,
    deep_two_paths_bottleneck_compression=0.5,
    l_ratio=0.5,
    ab_ratio=0.5,
    max_mix_idx=10,
    max_mix_deep_two_paths_idx=-1,
    model_name='two_path_inception_v3'
    ):
    """
    Wrapper for cai.inception_v3.compiled_two_path_inception_v3 .
    """
    return cai.inception_v3.compiled_two_path_inception_v3(
        input_shape=input_shape,
        classes=classes,
        two_paths_partial_first_block=two_paths_partial_first_block,
        two_paths_first_block=two_paths_first_block,
        two_paths_second_block=two_paths_second_block,
        deep_two_paths=deep_two_paths,
        deep_two_paths_compression=deep_two_paths_compression,
        deep_two_paths_bottleneck_compression=deep_two_paths_bottleneck_compression,
        l_ratio=l_ratio,
        ab_ratio=ab_ratio,
        max_mix_idx=max_mix_idx,
        max_mix_deep_two_paths_idx=max_mix_deep_two_paths_idx,
        model_name=model_name)

def create_paths(last_tensor, compression, l2_decay, dropout_rate=0.0):
    """Builds a new path from a previous main branch.
    # Arguments
        last_tensor: input tensor.
        compression: float, compression rate at transition layers.
        l2_decay: float.
        dropout_rate: if bigger than zero, adds a dropout layer.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    last_tensor = keras.layers.Conv2D(int(keras.backend.int_shape(last_tensor)[bn_axis] * compression), 1,
                      use_bias=True, activation='relu',
                      kernel_regularizer=keras.regularizers.l2(l2_decay))(last_tensor)
    if (dropout_rate>0): last_tensor = keras.layers.Dropout(dropout_rate)(last_tensor)
    last_tensor = keras.layers.MaxPooling2D(2, strides=2)(last_tensor)
    return last_tensor

def two_path_inception_v3(
    include_top=True,
    weights=None, #'two_paths_plant_leafs'
    input_shape=(224,224,3),
    pooling=None,
    classes=1000,
    two_paths_partial_first_block=0,
    two_paths_first_block=False,
    two_paths_second_block=False,
    deep_two_paths=False,
    deep_two_paths_compression=0.655,
    deep_two_paths_bottleneck_compression=0.5,
    l_ratio=0.5,
    ab_ratio=0.5,
    max_mix_idx=10,
    max_mix_deep_two_paths_idx=-1,
    model_name='two_path_inception_v3', 
    **kwargs):
    """
    Wrapper for cai.inception_v3.two_path_inception_v3 .
    """
    return cai.inception_v3.two_path_inception_v3(
    include_top=include_top,
    weights=weights, #'two_paths_plant_leafs'
    input_shape=input_shape,
    pooling=pooling,
    classes=classes,
    two_paths_partial_first_block=two_paths_partial_first_block,
    two_paths_first_block=two_paths_first_block,
    two_paths_second_block=two_paths_second_block,
    deep_two_paths=deep_two_paths,
    deep_two_paths_compression=deep_two_paths_compression,
    deep_two_paths_bottleneck_compression=deep_two_paths_bottleneck_compression,
    l_ratio=l_ratio,
    ab_ratio=ab_ratio,
    max_mix_idx=max_mix_idx,
    max_mix_deep_two_paths_idx=max_mix_deep_two_paths_idx,
    model_name=model_name)
