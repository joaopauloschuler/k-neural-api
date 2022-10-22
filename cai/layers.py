import tensorflow
import tensorflow.keras.layers
import tensorflow.keras.regularizers
import cai.util
import math

# Constants
# You can find the D6 Diagram at:
# https://www.researchgate.net/figure/Graphical-representation-of-our-pointwise-convolution-replacement-At-the-left-a-classic_fig1_355214501
def D6_4ch(): return 15
def D6_8ch(): return 14
def D6_12ch(): return 40
def D6_16ch(): return 2
def D6_24ch(): return 41
def D6_32ch(): return 13
def D6_64ch(): return 24
def D6_128ch(): return 26

def D6v3_2ch(): return 46
def D6v3_4ch(): return 45
def D6v3_8ch(): return 44
def D6v3_12ch(): return 42
def D6v3_16ch(): return 32
def D6v3_24ch(): return 43
def D6v3_32ch(): return 33
def D6v3_64ch(): return 34
def D6v3_128ch(): return 35

# kT3 is just a grouped convolution with the same constraints as explained in the paper:
# https://www.researchgate.net/publication/355214501_Grouped_Pointwise_Convolutions_Significantly_Reduces_Parameters_in_EfficientNet
def kT3_16ch(): return 3
def kT3_32ch(): return 23
def kT3_64ch(): return 25
def kT3_128ch(): return 27

def kT3v3_4ch(): return 47
def kT3v3_8ch(): return 48
def kT3v3_16ch(): return 36
def kT3v3_32ch(): return 37
def kT3v3_64ch(): return 38
def kT3v3_128ch(): return 39


class CopyChannels(tensorflow.keras.layers.Layer):
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

class Negate(tensorflow.keras.layers.Layer):
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
    
    def get_config(self):
        # this is here to make the warning to disappear.
        base_config = super(Negate, self).get_config()
        return dict(list(base_config.items()))

class ConcatNegation(tensorflow.keras.layers.Layer):        
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
        return tensorflow.keras.layers.Concatenate(axis=3)([x, -x])

    def get_config(self):
        # this is here to make the warning to disappear.
        base_config = super(ConcatNegation, self).get_config()
        return dict(list(base_config.items()))

class InterleaveChannels(tensorflow.keras.layers.Layer):
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
        return tensorflow.keras.layers.Concatenate(axis=3)(
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

class SumIntoHalfChannels(tensorflow.keras.layers.Layer):
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

    def get_config(self):
        # this is here to make the warning to disappear.
        base_config = super(SumIntoHalfChannels, self).get_config()
        return dict(list(base_config.items()))

def GlobalAverageMaxPooling2D(previous_layer,  name=None):
    """
    Adds both global Average and Max poolings. This layers is known to speed up training.
    """
    if name is None: name='global_pool'
    return tensorflow.keras.layers.Concatenate(axis=1)([
      tensorflow.keras.layers.GlobalAveragePooling2D(name=name+'_avg')(previous_layer),
      tensorflow.keras.layers.GlobalMaxPooling2D(name=name+'_max')(previous_layer)
    ])

def FitChannelCountTo(last_tensor, next_channel_count, has_interleaving=False, channel_axis=3):
    """
    Forces the number of channels to fit a specific number of channels.
    The new number of channels must be bigger than the number of input channels.
    The number of channels is fitted by concatenating copies of existing channels.
    """
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
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
    last_tensor = tensorflow.keras.layers.Concatenate(axis=channel_axis)( output_copies )
    return last_tensor

def EnforceEvenChannelCount(last_tensor, channel_axis=3):
    """
    Enforces that the number of channels is even (divisible by 2).
    """
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    if (prev_layer_channel_count % 2 > 0):
        last_tensor = FitChannelCountTo(
            last_tensor,
            next_channel_count=prev_layer_channel_count+1,
            channel_axis=channel_axis)
    return last_tensor

def BinaryConvLayers(last_tensor, name, shape=(3, 3), conv_count=1, has_batch_norm=True, has_interleaving=False, activation='relu', channel_axis=3):
    last_tensor = EnforceEvenChannelCount(last_tensor)
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    for conv_cnt in range(conv_count):
        input_tensor = last_tensor
        if has_interleaving:
            last_tensor_interleaved = InterleaveChannels(step_size=2, name=name+"_i_"+str(conv_cnt))(last_tensor)
        else:
            last_tensor_interleaved = last_tensor
        x1 = tensorflow.keras.layers.Conv2D(prev_layer_channel_count//2, shape, padding='same', activation=None, name=name+"_a_"+str(conv_cnt), groups=prev_layer_channel_count//2)(last_tensor)
        x2 = tensorflow.keras.layers.Conv2D(prev_layer_channel_count//2, shape, padding='same', activation=None, name=name+"_b_"+str(conv_cnt), groups=prev_layer_channel_count//2)(last_tensor_interleaved)
        last_tensor = tensorflow.keras.layers.Concatenate(axis=channel_axis, name=name+"_conc_"+str(conv_cnt))([x1,x2])
        if has_batch_norm: last_tensor = tensorflow.keras.layers.BatchNormalization(axis=channel_axis, name=name+"_batch_"+str(conv_cnt))(last_tensor)
        if activation is not None: last_tensor = tensorflow.keras.layers.Activation(activation=activation, name=name+"_act_"+str(conv_cnt))(last_tensor)
        from_highway = tensorflow.keras.layers.DepthwiseConv2D(1, # kernel_size
            strides=1,
            padding='valid',
            use_bias=False,
            name=name + '_depth_'+str(conv_cnt))(input_tensor)
        last_tensor = tensorflow.keras.layers.add([from_highway, last_tensor], name=name+'_add'+str(conv_cnt))
        if has_batch_norm: last_tensor = tensorflow.keras.layers.BatchNormalization(axis=channel_axis)(last_tensor)
    return last_tensor

def BinaryPointwiseConvLayers(last_tensor, name, conv_count=1, has_batch_norm=True, has_interleaving=False, activation='relu', channel_axis=3):
    return BinaryConvLayers(last_tensor, name, shape=(1, 1), conv_count=conv_count, has_batch_norm=has_batch_norm, has_interleaving=has_interleaving,  activation=activation, channel_axis=channel_axis)

def BinaryCompressionLayer(last_tensor, name, has_batch_norm=True, activation='relu', channel_axis=3):
    last_tensor = EnforceEvenChannelCount(last_tensor)
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    last_tensor = tensorflow.keras.layers.Conv2D(prev_layer_channel_count//2, (1, 1), padding='same', activation=None, name=name+"_conv", groups=prev_layer_channel_count//2)(last_tensor)
    if has_batch_norm: last_tensor = tensorflow.keras.layers.BatchNormalization(axis=channel_axis, name=name+"_batch")(last_tensor)
    if activation is not None: last_tensor = tensorflow.keras.layers.Activation(activation=activation, name=name+"_act")(last_tensor)
    return last_tensor

def BinaryCompression(last_tensor, name, target_channel_count, has_batch_norm=True, activation='relu', channel_axis=3):
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    cnt = 0
    while (prev_layer_channel_count >= target_channel_count * 2):
        last_tensor = BinaryCompressionLayer(last_tensor, name=name+'_'+str(cnt), has_batch_norm=has_batch_norm, activation=activation, channel_axis=channel_axis)
        prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
        cnt = cnt + 1
    if prev_layer_channel_count > target_channel_count:
        last_tensor = FitChannelCountTo(last_tensor, next_channel_count=target_channel_count*2, channel_axis=channel_axis)
        last_tensor = BinaryCompressionLayer(last_tensor, name=name+'_'+str(cnt), has_batch_norm=has_batch_norm, activation=activation, channel_axis=channel_axis)
    return last_tensor

def GetChannelAxis():
    """This function returns the channel axis."""
    if tensorflow.keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    return channel_axis
    
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
    groups=0,
    kernel_initializer="glorot_uniform",
    kernel_regularizer=None
    ):
    """Practical Conv2D wrapper.
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
        kernel_initializer: this is a very big open question.
        kernel_regularizer: a conservative L2 may be a good idea.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if tensorflow.keras.backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    # groups parameter isn't available in older tensorflow implementations
    if (groups>1) :
        x = tensorflow.keras.layers.Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            groups=groups,
            name=conv_name,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(x)
    else:
        x = tensorflow.keras.layers.Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=conv_name,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(x)

    if (has_batch_norm): x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, scale=has_batch_scale, name=bn_name)(x)
    if activation is not None: x = tensorflow.keras.layers.Activation(activation=activation, name=name)(x)
    return x

def HardSigmoid(x):
    """
    This function implements a hard sigmoid like function.
    You can find more info at https://paperswithcode.com/method/hard-sigmoid .
    This implementation returns values from 0 to 6.
    """
    return tensorflow.keras.layers.ReLU( 6.0 )( x + 3.0 ) * ( 1.0 / 6.0 )

def HardSwish(x):
    """
    This function implements thet hard swish function.
    You can find more info at https://paperswithcode.com/method/hard-swish .
    """
    return tensorflow.keras.layers.Multiply()([tensorflow.keras.layers.Activation(HardSigmoid)(x), x])
    
def kConv2DType0(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same'):
    return conv2d_bn(last_tensor, filters, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, strides=(stride_size, stride_size), padding=padding)

def kConv2DType1(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same'):
    output_tensor = last_tensor
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
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

def kConv2DType2(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, min_channels_per_group=16, kernel_size=1, stride_size=1, padding='same'):
    """
    This ktype is composed by a grouped convolution followed by interleaving and another grouped comvolution with skip connection. This basic architecture can
    vary according to the input tensor and its parameters. This is the basic building block for the papers:
    https://www.researchgate.net/publication/360226228_Grouped_Pointwise_Convolutions_Reduce_Parameters_in_Convolutional_Neural_Networks
    https://www.researchgate.net/publication/355214501_Grouped_Pointwise_Convolutions_Significantly_Reduces_Parameters_in_EfficientNet
    """
    output_tensor = last_tensor
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    output_channel_count = filters
    max_acceptable_divisor = (prev_layer_channel_count//min_channels_per_group)
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
            output_tensor = tensorflow.keras.layers.add([output_tensor, compression_tensor], name=name+'_inter_group_add')
    else:
        #print ('Dismissed groups:', group_count, 'Input channels:', prev_layer_channel_count, 'Output Channels:', output_channel_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
    return output_tensor

def kConv2DType3(last_tensor,  filters=32,  channel_axis=3,  name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same', min_channels_per_group=16):
    """
    Same as Type 1 but without interleaving and extra convolution.
    """
    output_tensor = last_tensor
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    output_channel_count = filters
    max_acceptable_divisor = (prev_layer_channel_count//min_channels_per_group)
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
    This basic architecture can vary according to input tensor and function parameter. This implementation differs from type 2 as the skip connection isn't made accross the interleaving layer.
    In internal documentation, this is solution D10.
    """
    output_tensor = last_tensor
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
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
            output_tensor = tensorflow.keras.layers.add([output_tensor, compression_tensor], name=name+'_inter_group_add')
    else:
        #print ('Dismissed groups:', group_count, 'Input channels:', prev_layer_channel_count, 'Output Channels:', output_channel_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
    return output_tensor

def kConv2DType6(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same'):
    """
    In internal documentation, this is solution D8.
    """
    output_tensor = last_tensor
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
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
            output_tensor = tensorflow.keras.layers.Concatenate(axis=3, name=name+'_inter_group_concat')([output_tensor, compression_tensor])
    else:
        #print ('Dismissed groups:', group_count, 'Input channels:', prev_layer_channel_count, 'Output Channels:', output_channel_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
    return output_tensor

def kConv2DType7(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, bin_conv_count=4, kernel_size=1, stride_size=1, padding='same'):
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    if filters >= prev_layer_channel_count:
        if filters > prev_layer_channel_count: last_tensor = FitChannelCountTo(last_tensor, next_channel_count=filters, channel_axis=channel_axis)
        if (bin_conv_count>0): last_tensor = BinaryPointwiseConvLayers(last_tensor, name, conv_count=bin_conv_count, activation=activation, has_batch_norm=has_batch_norm, channel_axis=channel_axis)
    if filters < prev_layer_channel_count:
        if (bin_conv_count>0): last_tensor = BinaryPointwiseConvLayers(last_tensor, name=name+'_biconv', conv_count=bin_conv_count, activation=activation, has_batch_norm=has_batch_norm, channel_axis=channel_axis)
        last_tensor = BinaryCompression(last_tensor, name=name+'_bicompress', activation=activation, has_batch_norm=has_batch_norm, target_channel_count=filters)
    return last_tensor

def kConv2DType8(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, min_channels_per_group=16, kernel_size=1, stride_size=1, padding='same', always_intergroup=False):
    """
    Same as Type 2 but with a different grouping for the second convolution.
    It's made by a grouped convolution followed by interleaving and another grouped comvolution with skip connection. This basic architecture can
    vary according to input tensor and function parameter. In internal documentation, this is solution D6.
    """
    output_tensor = last_tensor
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    output_channel_count = filters
    max_acceptable_divisor = (prev_layer_channel_count//min_channels_per_group)
    group_count = cai.util.get_max_acceptable_common_divisor(prev_layer_channel_count, output_channel_count, max_acceptable = max_acceptable_divisor)
    if group_count is None: group_count=1
    # input_group_size = prev_layer_channel_count // group_count
    if (group_count>1):
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias, strides=(stride_size, stride_size), padding=padding)
        compression_tensor = output_tensor
        second_conv_max_acceptable_divisor = (output_channel_count//min_channels_per_group)
        second_conv_group_count = cai.util.get_max_acceptable_common_divisor(output_channel_count, output_channel_count, max_acceptable = second_conv_max_acceptable_divisor)
        output_group_size = output_channel_count // second_conv_group_count
        #print ('Input channels:', prev_layer_channel_count, 'Output Channels:',output_channel_count,'Groups:', group_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        #print ('Second conv max acceptable divisor:', second_conv_max_acceptable_divisor, 'Output group count:',second_conv_group_count,'Second conv max acceptable divisor:', second_conv_max_acceptable_divisor)
        if second_conv_group_count > 1:
            if (prev_layer_channel_count >= output_channel_count) or (always_intergroup):
                #print('Has grouped intergroup')
                if activation is None: output_tensor = tensorflow.keras.layers.Activation(HardSwish)(output_tensor)
                output_tensor = InterleaveChannels(output_group_size, name=name+'_group_interleaved')(output_tensor)
                output_tensor = conv2d_bn(output_tensor, output_channel_count, 1, 1, name=name+'_group_interconn', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=second_conv_group_count, use_bias=use_bias)
        else:
            #print('Has intergroup')
            if activation is None: output_tensor = tensorflow.keras.layers.Activation(HardSwish)(output_tensor)
            output_tensor = conv2d_bn(output_tensor, output_channel_count, 1, 1, name=name+'_group_interconn', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
        
        # # Scales each channel.
        # output_tensor = tensorflow.keras.layers.DepthwiseConv2D((1, 1), use_bias=False, name=name+'_out_sep')(output_tensor)
        # compression_tensor = tensorflow.keras.layers.DepthwiseConv2D((1, 1), use_bias=False, name=name+'_out_comp')(compression_tensor)
        output_tensor = tensorflow.keras.layers.add([output_tensor, compression_tensor], name=name+'_inter_group_add')
    else:
        #print ('Dismissed groups:', group_count, 'Input channels:', prev_layer_channel_count, 'Output Channels:', output_channel_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
    return output_tensor

def kConv2DType9(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, min_channels_per_group=16, kernel_size=1, stride_size=1, padding='same', always_intergroup=False):
    """
    Same as Type 2 but with a different grouping for the second convolution.
    It's made by a grouped convolution followed by interleaving and another grouped comvolution with skip connection. This basic architecture can
    vary according to input tensor and function parameter. In internal documentation, this is solution D6.
    """
    output_tensor = last_tensor
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    output_channel_count = filters
    first_conv_min_group_channel_count = max(min_channels_per_group, int(math.sqrt(prev_layer_channel_count)) )
    max_acceptable_divisor = (prev_layer_channel_count//first_conv_min_group_channel_count)
    group_count = cai.util.get_max_acceptable_common_divisor(prev_layer_channel_count, output_channel_count, max_acceptable = max_acceptable_divisor)
    if group_count is None: group_count=1
    # input_group_size = prev_layer_channel_count // group_count
    if (group_count>1):
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias, strides=(stride_size, stride_size), padding=padding)
        compression_tensor = output_tensor
        second_conv_min_group_channel_count = max(min_channels_per_group, int(math.sqrt(output_channel_count)) )
        second_conv_max_acceptable_divisor = (output_channel_count//second_conv_min_group_channel_count)
        second_conv_group_count = cai.util.get_max_acceptable_common_divisor(output_channel_count, output_channel_count, max_acceptable = second_conv_max_acceptable_divisor)
        output_group_size = output_channel_count // second_conv_group_count
        #print ('Second conv max acceptable divisor:', second_conv_max_acceptable_divisor, 'Output group count:',second_conv_group_count,'Second conv max acceptable divisor:', second_conv_max_acceptable_divisor)
        #print ('Input channels:', prev_layer_channel_count, 'Output Channels:',output_channel_count,'Groups:', group_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        if second_conv_group_count > 1:
            if (prev_layer_channel_count >= output_channel_count) or (always_intergroup):
                #print('Has grouped intergroup')
                if activation is None: output_tensor = tensorflow.keras.layers.Activation(HardSwish)(output_tensor)
                output_tensor = InterleaveChannels(output_group_size, name=name+'_group_interleaved')(output_tensor)
                output_tensor = conv2d_bn(output_tensor, output_channel_count, 1, 1, name=name+'_group_interconn', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=second_conv_group_count, use_bias=use_bias)
        else:
            #print('Has intergroup')
            if activation is None: output_tensor = tensorflow.keras.layers.Activation(HardSwish)(output_tensor)
            output_tensor = conv2d_bn(output_tensor, output_channel_count, 1, 1, name=name+'_group_interconn', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
        
        # Scales each channel.
        #output_tensor = tensorflow.keras.layers.DepthwiseConv2D((1, 1), use_bias=False, name=name+'_out_sep')(output_tensor)
        #compression_tensor = tensorflow.keras.layers.DepthwiseConv2D((1, 1), use_bias=False, name=name+'_out_comp')(compression_tensor)
        output_tensor = tensorflow.keras.layers.add([output_tensor, compression_tensor], name=name+'_inter_group_add')
    else:
        #print ('Dismissed groups:', group_count, 'Input channels:', prev_layer_channel_count, 'Output Channels:', output_channel_count, 'Input channels per group:', input_group_size, 'Output channels per group:', output_group_size)
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
    return output_tensor
    
def kGroupConv2D(last_tensor, filters=32, channel_axis=3, channels_per_group=16, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same'):
    """ 
    This is a grouped convolution wrapper that tries to force the number of input channels per group. You can give any number of filters and groups.
    You can also add this layer after any layer with any number of channels independently on any common divisor requirement.
    Follows an example:
        * 1020 input channels.
        * 16 channels per group.
        * 250 filters.
    This is how  kGroupConv2D works:
        * The first step is to make the number of "input channels" multiple of the "number of input channels per group". So, we'll add 4 channels to the input by copying the first 4 channels. The total number of channels will be 1024.
        * The number of groups will be 1024/16 = 64 groups with 16 input channels each.
        * 250 filters aren't divisible by 64 groups. 250 mod 64 = 58. To solve this problem, we'll create 2 paths. The first path deals with the integer division while the second path deals with the remainder (modulo).
                Path 1: 250 filters divided by 64 groups equals 3 filters per group (integer division). So, the first path has a grouped convolution with 64 groups, 16 input channels per group and 3 filters per group. The total number of filters in this path is 64*3 = 192.
                Path 2: the remaining 58 filters are included in this second path. There will be 58 groups with 1 filter each. The first 58 * 16 = 928 channels will be copied and made as input layer for this path.
        * Both paths are then concatenated. As a result, we'll have 192 + 58 = 250 filers or output channels!
    """
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    groups = prev_layer_channel_count // channels_per_group
    if prev_layer_channel_count % channels_per_group > 0:
        groups = groups + 1
    # the number of groups should never be bigger than the number of filters.
    if groups > filters:
        groups = filters
    local_channels_per_group = channels_per_group
    if groups > 1:
        # do we need to add more channels to make the number of imput channels multiple of channels_per_group?
        if groups * channels_per_group > prev_layer_channel_count:
            last_tensor = FitChannelCountTo(last_tensor, next_channel_count=groups * channels_per_group, has_interleaving=False, channel_axis=channel_axis)
        # if we have few filters, we might end needing less channels per group. This is the only case that we'll have more channels per group.
        if groups * channels_per_group < prev_layer_channel_count:
            local_channels_per_group = prev_layer_channel_count // groups
            if ( prev_layer_channel_count % groups > 0):
                local_channels_per_group = local_channels_per_group + 1
            if local_channels_per_group * groups > prev_layer_channel_count:
                last_tensor = FitChannelCountTo(last_tensor, next_channel_count=groups * local_channels_per_group, has_interleaving=False, channel_axis=channel_axis)
        prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
        extra_filters = filters % groups
        # should we create an additional path so we can fit the extra filters?
        if (extra_filters == 0):
            last_tensor = conv2d_bn(last_tensor, filters-extra_filters, kernel_size, kernel_size, name=name+'_m'+str(groups), activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, padding=padding, strides=(stride_size, stride_size), groups=groups)
        else:
            root = last_tensor
            # the main path
            path1 = conv2d_bn(root, filters-extra_filters, kernel_size, kernel_size, name=name+'_p1_'+str(groups), activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, padding=padding, strides=(stride_size, stride_size), groups=groups)
            # we'll create one group per extra filter.
            path2 = CopyChannels(0, local_channels_per_group * extra_filters)(root)
            path2 = conv2d_bn(path2, extra_filters, kernel_size, kernel_size, name=name+'_p2', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, padding=padding, strides=(stride_size, stride_size), groups=extra_filters)
            # concats both paths.
            last_tensor =  tensorflow.keras.layers.Concatenate(axis=3, name=name+'_dc')([path1, path2]) # deep concat
    else:
        # deep unmodified.
        last_tensor = conv2d_bn(last_tensor, filters, kernel_size, kernel_size, name=name+'_dum', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, padding=padding, strides=(stride_size, stride_size))
    return last_tensor, groups

def kConv2DType10(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, min_channels_per_group=16, kernel_size=1, stride_size=1, padding='same', never_intergroup=False):
    """
    Same as Type 2 but with a different groupings. This is also a D6 type.
    It's made by a grouped convolution followed by interleaving and another grouped comvolution with skip connection.
    https://www.researchgate.net/figure/Graphical-representation-of-our-pointwise-convolution-replacement-At-the-left-a-classic_fig1_355214501
    """
    output_tensor = last_tensor
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    expansion = (filters > prev_layer_channel_count)
    # does it make sense to optimize this layer?
    if (prev_layer_channel_count > 2*min_channels_per_group) or (expansion and (prev_layer_channel_count > min_channels_per_group) ):
        output_tensor, group_count = kGroupConv2D(output_tensor, filters=filters, channel_axis=channel_axis, channels_per_group=min_channels_per_group, name=name+'_c1', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
        # should add a new convolution to mix group information?
        if (group_count>1) and (prev_layer_channel_count>=filters) and not(never_intergroup):
            compression_tensor = output_tensor
            # if activation is None: output_tensor = tensorflow.keras.layers.Activation(HardSwish)(output_tensor)
            interleave_step = filters // min_channels_per_group
            # should we interleave?
            if interleave_step>1: output_tensor = InterleaveChannels(interleave_step, name=name+'_i'+str(interleave_step))(output_tensor)
            output_tensor, group_count = kGroupConv2D(output_tensor, filters=filters, channel_axis=channel_axis, channels_per_group=min_channels_per_group, name=name+'_c2', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=1, stride_size=1, padding='valid')
            output_tensor = tensorflow.keras.layers.add([output_tensor, compression_tensor], name=name+'_iga')
    else:
        # unmofied
        output_tensor = conv2d_bn(last_tensor, filters, kernel_size, kernel_size, name=name+'_um', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, padding=padding, strides=(stride_size, stride_size))
    return output_tensor

def kConv2D(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same', kType=2):
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    if kType == 0:
        return kConv2DType0(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == 1:
        return kConv2DType1(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6_16ch():
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=16)
    elif kType == kT3_16ch():
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
        return kConv2DType7(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, bin_conv_count=5)
    elif kType == 12:
        return kConv2DType7(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, bin_conv_count=6)
    elif kType == D6_32ch():
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=32)
    elif kType == D6_8ch():
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=8)
    elif kType == D6_4ch():
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=4)
    elif kType == 16:
        if prev_layer_channel_count >= filters:
            return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=32)
        else:
            return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=16)
    elif kType == 17:
        if prev_layer_channel_count < filters:
            return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=32)
        else:
            return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=16)
    elif kType == 18:
        if prev_layer_channel_count >= filters:
            return kConv2DType7(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, bin_conv_count=5)
        else:
            return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=16)
    elif kType == 19:
        return kConv2DType8(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=16)
    elif kType == 20:
        return kConv2DType8(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=32)
    elif kType == 21:
        return kConv2DType8(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=16, always_intergroup=True)
    elif kType == 22:
        return kConv2DType8(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=32, always_intergroup=True)
    elif kType == kT3_32ch():
        return kConv2DType3(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=32)
    elif kType == D6_64ch():
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=64)
    elif kType == kT3_64ch():
        return kConv2DType3(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=64)
    elif kType == D6_128ch():
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=128)
    elif kType == kT3_128ch():
        return kConv2DType3(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=128)
    elif kType == 28:
        return kConv2DType9(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=16, always_intergroup=True)
    elif kType == 29:
        return kConv2DType9(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=32, always_intergroup=True)
    elif kType == 30:
        return kConv2DType9(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=64, always_intergroup=True)
    elif kType == 31:
        return kConv2DType9(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=128, always_intergroup=True)
    elif kType == D6v3_16ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=16, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_32ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=32, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_64ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=64, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_128ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=128, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == kT3v3_16ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=16, kernel_size=kernel_size, stride_size=stride_size, padding=padding, never_intergroup=True)
    elif kType == kT3v3_32ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=32, kernel_size=kernel_size, stride_size=stride_size, padding=padding, never_intergroup=True)
    elif kType == kT3v3_64ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=64, kernel_size=kernel_size, stride_size=stride_size, padding=padding, never_intergroup=True)
    elif kType == kT3v3_128ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=128, kernel_size=kernel_size, stride_size=stride_size, padding=padding, never_intergroup=True)
    elif kType == D6_12ch():
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=12)
    elif kType == D6_24ch():
        return kConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=24)
    elif kType == D6v3_12ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=12, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_24ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=24, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_8ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=8, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_4ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=4, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == D6v3_2ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=2, kernel_size=kernel_size, stride_size=stride_size, padding=padding)
    elif kType == kT3v3_4ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=4, kernel_size=kernel_size, stride_size=stride_size, padding=padding, never_intergroup=True)
    elif kType == kT3v3_8ch():
        return kConv2DType10(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, min_channels_per_group=8, kernel_size=kernel_size, stride_size=stride_size, padding=padding, never_intergroup=True)

def kPointwiseConv2D(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kType=2):
    """
    Parameter efficient pointwise convolution as shown in these papers:
    https://www.researchgate.net/publication/360226228_Grouped_Pointwise_Convolutions_Reduce_Parameters_in_Convolutional_Neural_Networks
    https://www.researchgate.net/publication/363413038_An_Enhanced_Scheme_for_Reducing_the_Complexity_of_Pointwise_Convolutions_in_CNNs_for_Image_Classification_Based_on_Interleaved_Grouped_Filters_without_Divisibility_Constraints
    """
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
        'SumIntoHalfChannels': SumIntoHalfChannels,
        'HardSigmoid': HardSigmoid,
        'HardSwish': HardSwish,
        'hard_sigmoid': HardSigmoid,
        'hard_swish': HardSwish
    }
