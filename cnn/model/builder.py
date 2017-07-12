# coding=utf-8
"""Module providing functionality for building convolutional neural nets."""

from collections import defaultdict

import tensorflow as tf


class CNNBuilder(object):
    """Implements logic to build a deep CNN.

    The CNNBuilder object is initiated by providing basic information about
    the network it is building. As the various methods are called,
    it keeps track of the current layer in the model, allowing it to easily
    build up a network, layer by layer.

    All functions must properly update the current top layer and its number
    of channels, as well as return them, in order for the model to work
    properly and be able to interface with custom created layers.
    """

    def __init__(self, input_layer, num_input_channels, is_train_phase,
                 weight_decay_rate, padding_mode, data_format, data_type):
        self.top_layer = input_layer
        self.num_top_channels = num_input_channels
        self.is_train_phase = is_train_phase
        self.weight_decay_rate = weight_decay_rate
        self.padding_mode = padding_mode
        self.data_format = data_format
        self.data_type = data_type
        self.layer_counts = defaultdict(int)

    def add_layer(self, layer, num_layer_channels):
        """Adds custom layer to model.

         In case a custom layer needed to be created without using one of
         the layer functions available in CNNBuilder, it is necessary to
         call this function afterwards, so that CNNBuilder can add the
         layer's information to its internal representation.

        Args:
            layer: Tensor. New layer to add to model.
            num_layer_channels: int. Number of channels in the layer.

        Returns:
            new_top_layer: Tensor. New top layer of model.
            new_num_top_channels: int. Number of channels in new top layer.
        """
        self.top_layer = layer
        self.num_top_channels = num_layer_channels
        return self.top_layer, self.num_top_channels

    def affine(self, num_out_channels, activation_method='relu',
               decay_kernel=True, decay_bias=True):
        """Adds a fully connected layer to the network.

        Args:
            num_out_channels: int. Number of output neurons.
            activation_method: string. Specifies which of the available
                               activation methods in _activation() to use.
            decay_kernel: bool. Whether weight decay regularization should be
                          applied to kernel variable.
            decay_bias: bool. Whether weight decay regularization should be
                        applied to bias variable.

        Returns:
            new_top_layer: Tensor. New top layer of model.
            new_num_top_channels: int. Number of channels in new top layer.
        """
        name = self._get_name('affine')
        with tf.variable_scope(name):
            # Compute fully connected layer with activation
            shape = [self.num_top_channels, num_out_channels]
            kernel_decay_rate = (self.weight_decay_rate if decay_kernel
                                 else 0.0)
            kernel = cnn_variable('weights', shape, data_type=self.data_type,
                                  weight_decay_rate=kernel_decay_rate)
            bias_decay_rate = self.weight_decay_rate if decay_bias else 0.0
            biases = cnn_variable('biases', [num_out_channels], 'zeros',
                                  self.data_type, bias_decay_rate)
            pre_activation = tf.matmul(self.top_layer, kernel) + biases
            affine = _get_activation(pre_activation, activation_method)

            # Add to activations collection for summary later
            tf.add_to_collection('activations', affine)

            self.top_layer = affine
            self.num_top_channels = num_out_channels
            return self.top_layer, self.num_top_channels

    def convolution(self, num_out_channels, kernel_height, kernel_width,
                    vertical_stride=1, horizontal_stride=1,
                    activation_method='relu', decay_kernel=True,
                    decay_bias=True, padding_mode=None):
        """Adds a convolutional layer to network.

        Args:
            num_out_channels: int. Number of channels in output.
            kernel_height: int. Height of filter used for convolution.
            kernel_width: int. Width of filter used for convolution.
            vertical_stride: int. Vertical stride.
            horizontal_stride: int. Horizontal stride.
            activation_method: string. Specifies which of the available
                               activation methods in _activation() to use.
            decay_kernel: bool. Whether weight decay regularization should be
                          applied to filter variable.
            decay_bias: bool. Whether weight decay regularization should be
                        applied to bias variable.
            padding_mode: 'SAME' or 'VALID'. Padding mode to use, or None for
                          CNNBuilder default


        Returns:
            new_top_layer: Tensor. New top layer of model.
            new_num_top_channels: int. Number of channels in new top layer.
        """
        name = self._get_name('conv')
        with tf.variable_scope(name):
            # Perform convolution , adjusted for proper data format, wit
            kernel_shape = [kernel_height, kernel_width, self.num_top_channels,
                            num_out_channels]
            kernel_decay_rate = (self.weight_decay_rate if decay_kernel
                                 else 0.0)
            kernel = cnn_variable('weights', kernel_shape,
                                  data_type=self.data_type,
                                  weight_decay_rate=kernel_decay_rate)
            if self.data_format == 'NHWC':
                strides = [1, vertical_stride, horizontal_stride, 1]
            else:
                strides = [1, 1, vertical_stride, horizontal_stride]
            padding_mode = padding_mode or self.padding_mode
            conv = tf.nn.conv2d(self.top_layer, kernel, strides, padding_mode,
                                data_format=self.data_format)
            # Add bias and apply activation
            bias_decay_rate = self.weight_decay_rate if decay_bias else 0.0
            biases = cnn_variable('biases', [num_out_channels], 'zeros',
                                  self.data_type, bias_decay_rate)
            pre_activation = tf.reshape(
                tf.nn.bias_add(conv, biases, self.data_format),
                conv.get_shape())
            conv1 = _get_activation(pre_activation, activation_method)
            # Add to activations collection for summary later
            tf.add_to_collection('activations', conv1)

            self.top_layer = conv1
            self.num_top_channels = num_out_channels
            return self.top_layer, self.num_top_channels

    def dropout(self, keep_prob=0.5):
        """Regularization dropout layer, only applied during training.

        Args:
            keep_prob: float. Probability that any independent neuron will
                       be dropped, if currently in training phase.

        Returns:
            new_top_layer: Tensor. New top layer of model.
            new_num_top_channels: int. Number of channels in new top layer.
        """
        name = self._get_name('dropout')
        # Only apply dropout during training
        if not self.is_train_phase:
            keep_prob = 1.0
        dropped = tf.nn.dropout(self.top_layer, keep_prob, name=name)
        self.top_layer = dropped
        return self.top_layer, self.num_top_channels

    def max_pooling(self, kernel_height, kernel_width, vertical_stride=2,
                    horizontal_stride=2, padding_mode=None):
        """Adds maximum pooling layer to network.

        Args:
            kernel_height: int. Height of window used for pooling.
            kernel_width: int. Width of window used for pooling.
            vertical_stride: int. Vertical stride.
            horizontal_stride: int. Horizontal stride.
            padding_mode: 'SAME' or 'VALID'. Padding mode to use, or None for
                          CNNBuilder default

        Returns:
            new_top_layer: Tensor. New top layer of model.
            new_num_top_channels: int. Number of channels in new top layer.
        """
        name = self._get_name('mpool')
        # Perform pooling, adjusted for the correct data format
        if self.data_format == 'NHWC':
            kernel_shape = [1, kernel_height, kernel_width, 1]
            strides = [1, vertical_stride, horizontal_stride, 1]
        else:
            kernel_shape = [1, 1, kernel_height, kernel_width]
            strides = [1, 1, vertical_stride, horizontal_stride]
        padding_mode = padding_mode or self.padding_mode
        pool = tf.nn.max_pool(self.top_layer, kernel_shape, strides,
                              padding_mode, self.data_format, name)

        self.top_layer = pool
        return self.top_layer, self.num_top_channels

    def normalization(self, depth_radius=None, bias=None, alpha=None,
                      beta=None):
        """Adds local response normalization layer to network.

        Check TensorFlow local_response_normalization() documentation for
        more information on the parameters.

        Args:
            depth_radius: float. Half-width of the 1-D normalization window.
            bias: float. An offset (usually positive to avoid dividing by 0).
            alpha: float. A scale factor, usually positive.
            beta: float. Defaults to 0.5. An exponent.

        Returns:
            new_top_layer: Tensor. New top layer of model.
            new_num_top_channels: int. Number of channels in new top layer.
        """
        name = self._get_name('norm')
        norm = tf.nn.local_response_normalization(self.top_layer, depth_radius,
                                                  bias, alpha, beta, name)
        self.top_layer = norm
        return self.top_layer, self.num_top_channels

    def reshape(self, shape):
        """Adds a reshape step to the network.

        Args:
            shape: list of ints. Shape to reshape to.

        Returns:
            new_top_layer: Tensor. New top layer of model.
            new_num_top_channels: int. Number of channels in new top layer.
        """
        name = self._get_name('reshape')
        reshaped = tf.reshape(self.top_layer, shape, name)
        new_shape = reshaped.get_shape()
        self.top_layer = reshaped
        if self.data_format == 'NHWC' or len(shape) < 4:
            self.num_top_channels = new_shape[-1].value
        else:
            self.num_top_channels = new_shape[-3].value
        return self.top_layer, self.num_top_channels

    def _get_name(self, prefix):
        """Creates unique name from prefix based on number of layers in CNN."""
        name = '{}{}'.format(prefix, self.layer_counts[prefix])
        self.layer_counts[prefix] += 1
        return name


def cnn_variable(name, shape, init_method='glorot_uniform',
                 data_type=tf.float32, weight_decay_rate=0.0):
    """Creates a variable on the CPU device, with options for initialization.

    Args:
        name: Name to use for variable.
        shape: Variable shape.
        init_method: string. Specifies which initialization method to use,
                     as available in _initializer().
        data_type: Data type of created variable.
        weight_decay_rate: bool. Decay rate for weights of variable, or 0.0 if
                           weight decay should not be applied.

    Returns:
        A variable placed on the CPU with the specified parameters.
    """
    if not data_type.is_floating:
        return TypeError('Variables must be initialized as floating point.')
    with tf.device('/cpu:0'):
        initializer = _get_initializer(shape, data_type, init_method)
        variable = tf.get_variable(name, shape, data_type, initializer)
    if weight_decay_rate:
        weight_decay = tf.multiply(tf.nn.l2_loss(variable), weight_decay_rate,
                                   '{}/decay_loss'.format(name))
        tf.add_to_collection('losses', weight_decay)
    return variable


def _get_activation(input_tensor, method):
    method = method or 'linear'
    return {
        'linear': tf.multiply(input_tensor, 1, 'linear'),
        'relu': tf.nn.relu(input_tensor),
        'sigmoid': tf.nn.sigmoid(input_tensor),
        'softmax': tf.nn.softmax(input_tensor),
        'tanh': tf.nn.tanh(input_tensor)
    }[method]


def _get_initializer(shape, data_type, method):
    # Determine number of input/output channels from shape
    fan_in = shape[-2] if len(shape) > 1 else shape[-1]
    fan_out = shape[-1]
    for dim in shape[:-2]:
        fan_in *= dim
        fan_out *= dim
    # Glorot uniform initialization helps improve variable training
    gu_val = (6 / (fan_in + fan_out)) ** 0.5

    method = method or 'zeros'
    return {
        'glorot_uniform': tf.random_uniform_initializer(-gu_val, gu_val,
                                                        dtype=data_type),
        'zeros': tf.zeros_initializer(data_type)
    }[method]
