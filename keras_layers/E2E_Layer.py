# -*- coding: utf-8 -*-
"""Custom Convolutional layers.
"""

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.engine.base_layer import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.layers import Add

class ConvE2E(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ConvE2E, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.common.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank,
                                                        'dilation_rate')

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        self.input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)

        # self.kernel = self.add_weight(shape=kernel_shape,
        #                               initializer=self.kernel_initializer,
        #                               name='kernel',
        #                               regularizer=self.kernel_regularizer,
        #                               constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: self.input_dim})
        self.built = True

    def call(self, inputs):
        kernel_h = self.kernel_size[0]
        kernel_w = self.kernel_size[1]

        kernel_size_h = conv_utils.normalize_tuple((kernel_h, 1), 2, 'kernel_size')
        kernel_shape_h = kernel_size_h + (self.input_dim, self.filters)
        kernel_dx1 = self.add_weight(shape=kernel_shape_h,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        kernel_size_w = conv_utils.normalize_tuple((1, kernel_w), 2, 'kernel_size')
        kernel_shape_w = kernel_size_w + (self.input_dim, self.filters)
        kernel_1xd = self.add_weight(shape=kernel_shape_w,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        outputs_dx1 = K.conv2d(inputs, kernel_dx1,strides=self.strides,
                               padding=self.padding, data_format=self.data_format,
                               dilation_rate=self.dilation_rate)
        outputs_dx1_dxd = K.repeat_elements(outputs_dx1, kernel_w, 1)

        outputs_1xd = K.conv2d(inputs, kernel_1xd, strides=self.strides,
                               padding=self.padding, data_format=self.data_format,
                               dilation_rate=self.dilation_rate)
        outputs_1xd_dxd = K.repeat_elements(outputs_1xd, kernel_h, 2)

        outputs = Add()([outputs_dx1_dxd, outputs_1xd_dxd])

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            output_shape = (input_shape) + (self.filters,)
            return output_shape
        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], self.filters) + (input_shape[1:])
            return output_shape

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ConvE2E, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
