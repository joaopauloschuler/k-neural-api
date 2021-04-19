from tensorflow import keras

class CopyChannels(keras.layers.Layer):
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
    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)
        self.trainable = False

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    
    def call(self, x):
        return -x

class ConcatNegation(keras.layers.Layer):        
    def __init__(self, **kwargs):
        super(ConcatNegation, self).__init__(**kwargs)
        self.trainable = False

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3]*2)
    
    def call(self, x):
        #return np.concatenate((x, -x), axis=3)
        return keras.layers.Concatenate(axis=3)([x, -x])

class InterleaveChannels(keras.layers.Layer):
    def __init__(self,
                 step_size=2,
                 **kwargs):
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

def GetClasses():
    """
    This function returns CAI layer classes.
    """
    return
    {
        'CopyChannels': CopyChannels,
        'Negate': Negate,
        'ConcatNegation': ConcatNegation,
        'InterleaveChannels': InterleaveChannels
    }
