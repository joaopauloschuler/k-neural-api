import tensorflow
from tensorflow import keras
import keras.layers
import cai.util

class CopyChannels(keras.layers.Layer):
    """
    This layer copies channels from channel_start the number of channels given in channel_count.
    """
    def __init__(self,
                 channel_start=0,
                 channel_count=1,
                 **kwargs):
        self.channel_start=channel_start
        self.channel_count=channel_count
        super(CopyChannels, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.channel_count)
    
    def call(self, x):
        return x[:, :, :, self.channel_start:(self.channel_start+self.channel_count)]
        
    def get_config(self):
        config = {
            'channel_start': self.channel_start,
            'channel_count': self.channel_count
        }
        base_config = super(CopyChannels, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Negate(keras.layers.Layer):
    """
    This layer negates (multiplies by -1) the input tensor.
    """
    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)
        self.trainable = False

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    
    def call(self, x):
        return -x

class ConcatNegation(keras.layers.Layer):        
    """
    This layer concatenates to the input its negation.
    """
    def __init__(self, **kwargs):
        super(ConcatNegation, self).__init__(**kwargs)
        self.trainable = False

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3]*2)
    
    def call(self, x):
        #return np.concatenate((x, -x), axis=3)
        return keras.layers.Concatenate(axis=3)([x, -x])

class InterleaveChannels(keras.layers.Layer):
    """
    This layer interleaves channels stepping according to the number passed as parameter.
    """
    def __init__(self,
                 step_size=2,
                 **kwargs):
        if step_size < 2:
            self.step_size=1
        else:
            self.step_size=step_size
        super(InterleaveChannels, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])

    def call(self, x):
        return keras.layers.Concatenate(axis=3)(
            [ x[:, :, :, shift_pos::self.step_size] for shift_pos in range(self.step_size) ]
        )
        # for self.step_size == 2, we would have:
        #  return keras.layers.Concatenate(axis=3)([
        #    x[:, :, :, 0::self.step_size],
        #    x[:, :, :, 1::self.step_size]
        #    ])

    def get_config(self):
        config = {
            'step_size': self.step_size
        }
        base_config = super(InterleaveChannels, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SumIntoHalfChannels(keras.layers.Layer):
    """
    This layer divedes channels into 2 halfs and then sums resulting in half of the input channels.
    """
    def __init__(self, **kwargs):
        super(SumIntoHalfChannels, self).__init__(**kwargs)
        self.trainable = False

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] // 2)

    def call(self, x):
        outputchannels = x.shape[3] // 2
        return tensorflow.math.add(
            x=x[:, :, :, 0:outputchannels],
            y=x[:, :, :, outputchannels:outputchannels*2]
            )

def GlobalAverageMaxPooling2D(previous_layer,  name=None):
    """
    Adds both global Average and Max poolings. This layers is known to speed up training.
    """
    if name is None: name='global_pool'
    return keras.layers.Concatenate(axis=1)([
      keras.layers.GlobalAveragePooling2D(name=name+'_avg')(previous_layer),
      keras.layers.GlobalMaxPooling2D(name=name+'_max')(previous_layer)
    ])

def FitChannelCountTo(last_tensor, next_channel_count, has_interleaving=False, channel_axis=3):
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    full_copies = next_channel_count // prev_layer_channel_count
    extra_channels = next_channel_count % prev_layer_channel_count
    output_copies = []
    for copy_cnt in range(full_copies):
        if copy_cnt == 0:
            output_copies.append( last_tensor )
        else:
            if has_interleaving:
                output_copies.append( InterleaveChannels(step_size=((copy_cnt+1) % prev_layer_channel_count))(last_tensor) )
            else:
                output_copies.append( last_tensor )
    if (extra_channels > 0):
        if has_interleaving:
            extra_tensor = InterleaveChannels(step_size=((full_copies+1) % prev_layer_channel_count))(last_tensor)
        else:
            extra_tensor = last_tensor
        output_copies.append( CopyChannels(0,extra_channels)(extra_tensor) )
    last_tensor = keras.layers.Concatenate(axis=channel_axis)( output_copies )
    return last_tensor

def EnforceEvenChannelCount(last_tensor, channel_axis=3):
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    if (prev_layer_channel_count % 2 > 0):
        last_tensor = FitChannelCountTo(
            last_tensor,
            next_channel_count=prev_layer_channel_count+1,
            channel_axis=channel_axis)
    return last_tensor

def BinaryConvLayers(last_tensor, name, shape=(3, 3), conv_count=1, has_batch_norm=True, has_interleaving=False, activation='relu', channel_axis=3):
    last_tensor = EnforceEvenChannelCount(last_tensor)
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    for conv_cnt in range(conv_count):
        input_tensor = last_tensor
        if has_interleaving:
            last_tensor_interleaved = InterleaveChannels(step_size=2, name=name+"_i_"+str(conv_cnt))(last_tensor)
        else:
            last_tensor_interleaved = last_tensor
        x1 = keras.layers.Conv2D(prev_layer_channel_count//2, shape, padding='same', activation=None, name=name+"_a_"+str(conv_cnt), groups=prev_layer_channel_count//2)(last_tensor)
        x2 = keras.layers.Conv2D(prev_layer_channel_count//2, shape, padding='same', activation=None, name=name+"_b_"+str(conv_cnt), groups=prev_layer_channel_count//2)(last_tensor_interleaved)
        last_tensor = keras.layers.Concatenate(axis=channel_axis, name=name+"_conc_"+str(conv_cnt))([x1,x2])
        if has_batch_norm: last_tensor = keras.layers.BatchNormalization(axis=channel_axis, name=name+"_batch_"+str(conv_cnt))(last_tensor)
        if activation is not None: last_tensor = keras.layers.Activation(activation=activation, name=name+"_act_"+str(conv_cnt))(last_tensor)
        last_tensor = keras.layers.add([input_tensor, last_tensor], name=name+'_add'+str(conv_cnt))
        if has_batch_norm: last_tensor = keras.layers.BatchNormalization(axis=channel_axis)(last_tensor)
    return last_tensor

def BinaryPointwiseConvLayers(last_tensor, name, conv_count=1, has_batch_norm=True, has_interleaving=False, activation='relu', channel_axis=3):
    return BinaryConvLayers(last_tensor, name, shape=(1, 1), conv_count=conv_count, has_batch_norm=has_batch_norm, has_interleaving=has_interleaving,  activation=activation, channel_axis=channel_axis)

def BinaryCompressionLayer(last_tensor, name, has_batch_norm=True, activation='relu', channel_axis=3):
    last_tensor = EnforceEvenChannelCount(last_tensor)
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    last_tensor = keras.layers.Conv2D(prev_layer_channel_count//2, (1, 1), padding='same', activation=None, name=name+"_conv", groups=prev_layer_channel_count//2)(last_tensor)
    if has_batch_norm: last_tensor = keras.layers.BatchNormalization(axis=channel_axis, name=name+"_batch")(last_tensor)
    if activation is not None: last_tensor = keras.layers.Activation(activation=activation, name=name+"_act")(last_tensor)
    return last_tensor

def BinaryCompression(last_tensor, name, target_channel_count, has_batch_norm=True, activation='relu', channel_axis=3):
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    cnt = 0
    while (prev_layer_channel_count >= target_channel_count * 2):
        last_tensor = BinaryCompressionLayer(last_tensor, name=name+'_'+str(cnt), has_batch_norm=has_batch_norm, activation=activation, channel_axis=channel_axis)
        prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
        cnt = cnt + 1
    if prev_layer_channel_count > target_channel_count:
        last_tensor = FitChannelCountTo(last_tensor, next_channel_count=target_channel_count*2, channel_axis=channel_axis)
        last_tensor = BinaryCompressionLayer(last_tensor, name=name+'_'+str(cnt), has_batch_norm=has_batch_norm, activation=activation, channel_axis=channel_axis)
    return last_tensor

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              use_bias=False,
              activation='relu', 
              has_batch_norm=True,
              has_batch_scale=False,  
              groups=0
              ):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
        use_bias: True means that bias will be added,
        activation: activation function. None means no activation function. 
        has_batch_norm: True means that batch normalization is added.
        has_batch_scale: True means that scaling is added to batch norm.
        groups: number of groups in the convolution

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if keras.backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = keras.layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        groups=groups, 
        name=conv_name)(x)
    if (has_batch_norm): x = keras.layers.BatchNormalization(axis=bn_axis, scale=has_batch_scale, name=bn_name)(x)
    if activation is not None: x = keras.layers.Activation(activation=activation, name=name)(x)
    return x
    
def kConv2DType0(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same'):
    return conv2d_bn(last_tensor, filters, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, strides=(stride_size, stride_size), padding=padding)

def kConv2DType1(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same'):
    output_tensor = last_tensor
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    output_channel_count = filters
    max_acceptable_divisor = (prev_layer_channel_count//16)
    group_count = cai.util.get_max_acceptable_common_divisor(prev_layer_channel_count, output_channel_count, max_acceptable = max_acceptable_divisor)
    if group_count is None: group_count=1
    output_group_size = output_channel_count // group_count
    # input_group_size = prev_layer_channel_count // group_count
    if (group_count > 1):
        #print ('Input channels:', prev_layer_channel_count, 'Output Channels:',output_channel_count,'Groups:', group_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias, strides=(stride_size, stride_size), padding=padding)
        if output_group_size > 1:
          output_tensor = InterleaveChannels(output_group_size, name=name+'_group_interleaved')(output_tensor)
    else:
        #print ('Dismissed groups:', group_count, 'Input channels:', prev_layer_channel_count, 'Output Channels:', output_channel_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)        
    return output_tensor

def kConv2DType2(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, max_channels_per_group=16, kernel_size=1, stride_size=1, padding='same'):
    """
    This is the default ktype. It's made by a grouped convolution followed by interleaving and another grouped comvolution with skip connection. This basic architecture can
    vary according to input tensor and function parameter. In internal documentation, this is solution D6.
    """
    output_tensor = last_tensor
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    output_channel_count = filters
    max_acceptable_divisor = (prev_layer_channel_count//max_channels_per_group)
    group_count = cai.util.get_max_acceptable_common_divisor(prev_layer_channel_count, output_channel_count, max_acceptable = max_acceptable_divisor)
    if group_count is None: group_count=1
    output_group_size = output_channel_count // group_count
    # input_group_size = prev_layer_channel_count // group_count
    if (group_count>1):
        #print ('Input channels:', prev_layer_channel_count, 'Output Channels:',output_channel_count,'Groups:', group_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias, strides=(stride_size, stride_size), padding=padding)
        compression_tensor = output_tensor
        if output_group_size > 1:
            output_tensor = InterleaveChannels(output_group_size, name=name+'_group_interleaved')(output_tensor)
        if (prev_layer_channel_count >= output_channel_count):
            #print('Has intergroup')
            output_tensor = conv2d_bn(output_tensor, output_channel_count, 1, 1, name=name+'_group_interconn', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias)
            output_tensor = keras.layers.add([output_tensor, compression_tensor], name=name+'_inter_group_add')
    else:
        #print ('Dismissed groups:', group_count, 'Input channels:', prev_layer_channel_count, 'Output Channels:', output_channel_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
    return output_tensor

def kConv2DType3(last_tensor,  filters=32,  channel_axis=3,  name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same'):
    """
    Same as Type 1 but without interleaving and extra convolution.
    """
    output_tensor = last_tensor
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    output_channel_count = filters
    max_acceptable_divisor = (prev_layer_channel_count//16)
    group_count = cai.util.get_max_acceptable_common_divisor(prev_layer_channel_count, output_channel_count, max_acceptable = max_acceptable_divisor)
    if group_count is None: group_count=1
    # output_group_size = output_channel_count // group_count
    # input_group_size = prev_layer_channel_count // group_count
    if (group_count > 1):
        #print ('Input channels:', prev_layer_channel_count, 'Output Channels:',output_channel_count,'Groups:', group_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias, strides=(stride_size, stride_size), padding=padding)
    else:
        #print ('Dismissed groups:', group_count, 'Input channels:', prev_layer_channel_count, 'Output Channels:', output_channel_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, strides=(stride_size, stride_size), padding=padding)
    return output_tensor

def kConv2DType4(last_tensor,  filters=32,  channel_axis=3,  name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same'):
    """
    This is Type 2 followed by Type 3.
    """
    last_tensor  = kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    last_tensor  = kConv2DType3(last_tensor, filters=filters, channel_axis=channel_axis, name='e_'+name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    return last_tensor

def kConv2DType5(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same'):
    """
    This is the default ktype. It's made by a grouped convolution followed by interleaving and another grouped comvolution with skip connection. This basic architecture can
    vary according to input tensor and function parameter. This implementation differs from type 2 as the skip connection isn't made accross the interleaving layer.
    In internal documentation, this is solution D10.
    """
    output_tensor = last_tensor
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    output_channel_count = filters
    max_acceptable_divisor = (prev_layer_channel_count//16)
    group_count = cai.util.get_max_acceptable_common_divisor(prev_layer_channel_count, output_channel_count, max_acceptable = max_acceptable_divisor)
    if group_count is None: group_count=1
    output_group_size = output_channel_count // group_count
    # input_group_size = prev_layer_channel_count // group_count
    if (group_count>1):
        #print ('Input channels:', prev_layer_channel_count, 'Output Channels:',output_channel_count,'Groups:', group_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias, strides=(stride_size, stride_size), padding=padding)
        if output_group_size > 1:
            output_tensor = InterleaveChannels(output_group_size, name=name+'_group_interleaved')(output_tensor)
        if (prev_layer_channel_count >= output_channel_count):
            #print('Has intergroup')
            compression_tensor = output_tensor
            output_tensor = conv2d_bn(output_tensor, output_channel_count, 1, 1, name=name+'_group_interconn', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias)
            output_tensor = keras.layers.add([output_tensor, compression_tensor], name=name+'_inter_group_add')
    else:
        #print ('Dismissed groups:', group_count, 'Input channels:', prev_layer_channel_count, 'Output Channels:', output_channel_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
    return output_tensor

def kConv2DType6(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same'):
    """
    In internal documentation, this is solution D8.
    """
    output_tensor = last_tensor
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    output_channel_count = filters
    max_acceptable_divisor = (prev_layer_channel_count//16)
    group_count = cai.util.get_max_acceptable_common_divisor(prev_layer_channel_count, output_channel_count, max_acceptable = max_acceptable_divisor)
    if group_count is None: group_count=1
    output_group_size = output_channel_count // group_count
    # input_group_size = prev_layer_channel_count // group_count
    if (group_count>1):
        #print ('Input channels:', prev_layer_channel_count, 'Output Channels:',output_channel_count,'Groups:', group_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias, strides=(stride_size, stride_size), padding=padding)
        compression_tensor = output_tensor
        compression_tensor = SumIntoHalfChannels()(compression_tensor)
        if output_group_size > 1:
            output_tensor = InterleaveChannels(output_group_size, name=name+'_group_interleaved')(output_tensor)
        if (prev_layer_channel_count >= output_channel_count):
            #print('Has intergroup')
            output_tensor = conv2d_bn(output_tensor, output_channel_count, 1, 1, name=name+'_group_interconn', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias)
            output_tensor = SumIntoHalfChannels()(output_tensor)
            output_tensor = keras.layers.Concatenate(axis=3, name=name+'_inter_group_concat')([output_tensor, compression_tensor])
    else:
        #print ('Dismissed groups:', group_count, 'Input channels:', prev_layer_channel_count, 'Output Channels:', output_channel_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
    return output_tensor

def kConv2DType7(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, bin_conv_count=4, kernel_size=1, stride_size=1, padding='same'):
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    if filters >= prev_layer_channel_count:
        if filters > prev_layer_channel_count: last_tensor = FitChannelCountTo(last_tensor, next_channel_count=filters, channel_axis=channel_axis)
        if (bin_conv_count>0): last_tensor = BinaryPointwiseConvLayers(last_tensor, name, conv_count=bin_conv_count, activation=activation, has_batch_norm=has_batch_norm, channel_axis=channel_axis)
    if filters < prev_layer_channel_count:
        if (bin_conv_count>0): last_tensor = BinaryPointwiseConvLayers(last_tensor, name=name+'_biconv', conv_count=bin_conv_count, activation=activation, has_batch_norm=has_batch_norm, channel_axis=channel_axis)
        last_tensor = BinaryCompression(last_tensor, name=name+'_bicompress', activation=activation, has_batch_norm=has_batch_norm, target_channel_count=filters)
    return last_tensor

def kConv2D(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same', kType=2):
    if kType == 0:
        return kConv2DType0(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == 1:
        return kConv2DType1(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == 2:
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, max_channels_per_group=16)
    elif kType == 3:
        return kConv2DType3(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == 4:
        return kConv2DType4(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == 5:
        return kConv2DType5(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == 6:
        return kConv2DType6(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == 7:
        return kConv2DType7(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, bin_conv_count=0)
    elif kType == 8:
        return kConv2DType7(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, bin_conv_count=1)
    elif kType == 9:
        return kConv2DType7(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, bin_conv_count=2)
    elif kType == 10:
        return kConv2DType7(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, bin_conv_count=4)
    elif kType == 11:
        return kConv2DType7(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, bin_conv_count=8)
    elif kType == 12:
        return kConv2DType7(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, bin_conv_count=16)
    elif kType == 13:
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, max_channels_per_group=32)
    elif kType == 14:
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, max_channels_per_group=8)
    elif kType == 15:
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, max_channels_per_group=4)

def kPointwiseConv2D(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kType=2):
    return kConv2D(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=1, stride_size=1, padding='same', kType=kType)

def GetClasses():
    """
    This function returns CAI layer classes.
    """
    return {
        'CopyChannels': CopyChannels,
        'Negate': Negate,
        'ConcatNegation': ConcatNegation,
        'InterleaveChannels': InterleaveChannels,
        'SumIntoHalfChannels': SumIntoHalfChannels
    }
