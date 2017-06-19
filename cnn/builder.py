# coding=utf-8
"""Module provides ability to easily create convolutional neural networks"""

import tensorflow as tf
from collections import defaultdict


def cnn_variable(name, shape, use_xavier_init=True, data_type=tf.float32):
    """Creates a variable on the CPU device, with options for initialization.

    Args:
        name: Name to use for variable.
        shape: Variable shape.
        use_xavier_init: bool. If True, uses Xavier method for initializing
                         the weights to improve network performance.
                         Otherwise initializes values to 0.
        data_type: Data type in output variable.

    Returns:
        A variable placed on the CPU with the specified parameters.
    """
    if not data_type.is_floating:
        return TypeError('Variables must be initialized as floating point.')
    with tf.device('/cpu:0'):
        n_in = shape[-2] if len(shape) > 1 else shape[-1]
        n_out = shape[-1]
        for dim in shape[:-2]:
            n_in *= dim
            n_out *= dim
        n_avg = (n_in + n_out) / 2.0
        lim = (3.0 / n_avg) ** 0.5
        if use_xavier_init:
            initializer = tf.random_uniform_initializer(-lim, lim)
        else:
            initializer = tf.zeros_initializer()
        variable = tf.get_variable(name, shape, data_type, initializer)
    return variable


class CNNBuilder(object):
    """Implements logic to connect multiple layers of network together"""

    def __init__(self, input_op, input_channels, training_phase, mode='SAME',
                 data_format='NCHW', data_type=tf.float32):
        self.top_layer = input_op
        self.top_channels = input_channels
        self.training_phase = training_phase
        self.mode = mode
        self.data_format = data_format
        self.data_type = data_type
        self.layer_counts = defaultdict(int)

    def convolution(self, out_channels, filter_height, filter_width,
                    vertical_stride=1, horizontal_stride=1, use_relu=True):
        """Adds convolutional layer to network.

         Weights for filter initialized using Xavier method, biases
         initialized to zero, and there is an optional activation using ReLU
         following convolution.

        Args:
            out_channels: int. Number of channels in output.
            filter_height: int. Height of filter used for convolution.
            filter_width: int. Width of filter used for convolution.
            vertical_stride: int. Vertical stride.
            horizontal_stride: int. Horizontal stride.
            use_relu: bool. If True, use ReLU activation following convolution.
        Returns:
            Tensor output of convolution layer.
        """
        name = 'conv{0:d}'.format(self.layer_counts['conv'])
        self.layer_counts['conv'] += 1
        with tf.variable_scope(name):
            filter_shape = [filter_height, filter_width, self.top_channels,
                            out_channels]
            filter = cnn_variable('filter', filter_shape, True, self.data_type)
            strides = [1, vertical_stride, horizontal_stride, 1]
            if self.data_format == 'NCHW':
                strides = [strides[0], strides[3], strides[1], strides[2]]
            conv = tf.nn.conv2d(self.top_layer, filter, strides, self.mode,
                                data_format=self.data_format)
            biases = cnn_variable('biases', [out_channels], False,
                                  self.data_type)
            pre_activation = tf.nn.bias_add(conv, biases, self.data_format)
            conv1 = tf.nn.relu(pre_activation) if use_relu else pre_activation
            self.top_layer = conv1
            self.top_channels = out_channels
            return conv1

    def dropout(self, keep_prob=0.5):
        """Regularization dropout layer, only applied during training.

        Args:
            keep_prob: float. Probability that any independent neuron will
                       be dropped, if currently in training phase.

        Returns:
            Tensor output from dropout layer.
        """
        name = 'dropout{0:d}'.format(self.layer_counts['dropout'])
        self.layer_counts['dropout'] += 1
        if not self.training_phase:
            keep_prob = 1.0
        dropped = tf.nn.dropout(self.top_layer, keep_prob, name=name)
        self.top_layer = dropped
        return dropped

    def fully_connected(self, out_channels, use_relu=True):
        """Adds a fully connected layer to the network.

        Args:
            out_channels: int. Number of output neurons.
            use_relu: bool. If True, use ReLU activation after layer.

        Returns:
            Tensor output of fully connected layer.
        """
        name = 'fc{0:d}'.format(self.layer_counts['fc'])
        self.layer_counts['fc'] += 1
        with tf.variable_scope(name):
            shape = [self.top_channels, out_channels]
            weights = cnn_variable('weights', shape, True, self.data_type)
            biases = cnn_variable('biases', [out_channels], False,
                                  self.data_type)
            pre_activation = tf.matmul(self.top_layer, weights) + biases
            fc = tf.nn.relu(pre_activation) if use_relu else pre_activation
            self.top_layer = fc
            self.top_channels = out_channels
            return fc

    def max_pooling(self, pool_height, pool_width, vertical_stride=1,
                    horizontal_stride=2):
        """Adds maximum pooling layer to network.

        Args:
            pool_height: int. Height of window used for pooling.
            pool_width: int. Width of window used for pooling.
            vertical_stride: int. Vertical stride.
            horizontal_stride: int. Horizontal stride.
        Returns:
            Tensor output from max pooling  layer.
        """
        name = 'mpool{0:d}'.format(self.layer_counts['mpool'])
        self.layer_counts['mpool'] += 1
        window = [1, pool_height, pool_width, 1]
        strides = [1, vertical_stride, horizontal_stride, 1]
        pool = tf.nn.max_pool(self.top_layer, window, strides, self.mode,
                              self.data_format, name)
        self.top_layer = pool
        return pool

    def normalization(self, depth_radius=None, bias=None, alpha=None,
                      beta=None):
        """Adds local response normalization layer to network.

        Check TensorFlow local_response_normalization() documentation for
        default actions taken when the parameters are not provided.

        Args:
            depth_radius: float. Half-width of the 1-D
            bias: float. An offset (usually positive to avoid dividing by 0).
            alpha: float. A scale factor, usually positive.
            beta: float. Defaults to 0.5. An exponent.

        Returns:
            Tensor output from normalization layer
        """
        name = 'norm{0:d}'.format(self.layer_counts['norm'])
        self.layer_counts['norm'] += 1
        norm = tf.nn.local_response_normalization(self.top_layer, depth_radius,
                                                  bias, alpha, beta, name)
        self.top_layer = norm
        return norm

    def reshape(self, shape):
        """Adds a reshape step to the network.

        Args:
            shape: list of ints. Shape to reshape to.

        Returns:
            The reshaped Tensor in the network.
        """
        reshaped = tf.reshape(self.top_layer)
        self.top_layer = reshaped
        self.top_channels = shape[-1]
        return self.top_layer
