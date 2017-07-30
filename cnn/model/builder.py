# coding=utf-8
"""Module providing functionality for building convolutional neural nets."""

from collections import defaultdict

import tensorflow as tf


# noinspection PyMissingOrEmptyDocstring
class CNNBuilder(object):
    """Implements logic to build a deep convolutional neural network.

    The methods in CNNBuilder provide a wrapper around the methods in
    tf.layer, making them easier to use by setting the correct parameters
    automatically based on the model configuration, and by keeping track of
    the current top layer in the model as it is built.
    """

    def __init__(self, input_layer, is_training, use_batch_norm,
                 weight_decay_rate, padding_mode, data_format):
        """Initialize information about network to be built.

        Args:
            input_layer: Tensor. Input layer (i.e. image) to feed into model.
            is_training: bool. Whether model is currently in training phase.
                         This affects layers like dropout and batch
                         normalization.
            use_batch_norm: bool. Whether batch normalization should be
                            applied after each convolutional and dense layer.
            weight_decay_rate: float. Rate to use for weight decay
                               regularization. 0 indicates no regularization
                               should be used
            padding_mode: 'SAME' or 'VALID'. Padding mode to use by default.
            data_format: 'NCHW' or 'NHWC'. Data format to use for Tensors.
        """
        self._top_layer = input_layer
        self._is_training = is_training
        self._use_batch_norm = use_batch_norm
        self._weight_decay_rate = weight_decay_rate
        self._padding_mode = padding_mode
        self._channel_pos = (data_format == 'NCHW' and 'channels_first' or
                             data_format == 'NHWC' and 'channels_last')
        self._layer_counts = defaultdict(int)

    @property
    def top_layer(self):
        return self._top_layer

    @top_layer.setter
    def top_layer(self, layer):
        """As CNNBuilder automatically updates its top layer as the methods
        are called, this should only be used in case a custom layer has been
        created without CNNBuilder methods, and needs to be added to the
        model. """
        self._top_layer = layer

    @property
    def is_training(self):
        return self._is_training

    @property
    def use_batch_norm(self):
        return self._use_batch_norm

    @property
    def weight_decay_rate(self):
        return self._weight_decay_rate

    @property
    def padding_mode(self):
        return self._padding_mode

    def average_pooling(self, pool_size, strides=1, padding_mode=None):
        """Adds average pooling layer to network.

        Args:
            pool_size: int or (int, int). Height and width of 2D pooling
                       window. Single integer indicates height and width are
                       same size.
            strides: int or (int, int). Vertical and horizontal stride of
                     pooling window. Single integer indicates strides in both
                     dimensions are of same size.
            padding_mode: 'same' or 'valid'. Padding mode to use. None to use
                          CNNBuilder object's default.
        """
        name = self._get_layer_name('avgpool')
        padding_mode = padding_mode or self._padding_mode
        avg_pool = tf.layers.average_pooling2d(
            self._top_layer, pool_size, strides, padding_mode,
            self._channel_pos, name)
        self._top_layer = avg_pool

    def batch_normalization(self):
        """Adds batch normalization layer to network. Note that this is
        already called automatically by convolutional and dense layers, so be
        careful not to call it twice in a row."""
        name = self._get_layer_name('batchnorm')
        axis = 1 if self._channel_pos == 'channels_first' else -1
        batch_norm = tf.layers.batch_normalization(
            self._top_layer, axis, training=self._is_training, name=name)
        self._top_layer = batch_norm

    def convolution(self, num_filters, kernel_size, strides=1,
                    activation_method='relu', use_batch_norm=None,
                    padding_mode=None):
        """Adds a convolutional layer to network.

        Args:
            num_filters: int. Number of channels in output.
            kernel_size: int or (int, int). Height and width of 2D convolution
                         window. Single integer indicates height and width are
                         same size.
            strides: int or (int, int). Vertical and horizontal stride of
                     convolution window. Single integer indicates strides in
                     both dimensions are of same size.
            activation_method: string. Specifies which of the available
                               activations in _get_activation_func() to use.
            use_batch_norm: bool. Whether batch normalization should be
                            applied to this layer. None to use CNNBuilder
                            object's default.
            padding_mode: 'SAME' or 'VALID'. Padding mode to use. None to use
                          CNNBuilder object's default.
        """
        name = self._get_layer_name('conv')
        with tf.variable_scope(name):
            if padding_mode is None:
                padding_mode = self._padding_mode
            if use_batch_norm is None:
                use_batch_norm = self._use_batch_norm

            weight_decay_func = self._get_weight_decay_func()
            activation_func = (self._get_activation_func(activation_method) if
                               not use_batch_norm else None)
            use_bias = not use_batch_norm

            self._top_layer = tf.layers.conv2d(
                self._top_layer, num_filters, kernel_size, strides,
                padding_mode, self._channel_pos, activation=activation_func,
                use_bias=use_bias, kernel_regularizer=weight_decay_func,
                bias_regularizer=weight_decay_func)
            if use_batch_norm:
                self.batch_normalization()
                activation_func = self._get_activation_func(activation_method)
                self._top_layer = activation_func(self._top_layer)

            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self._top_layer)

    def dense(self, num_output_units, activation_method='relu',
              use_batch_norm=None):
        """Adds a fully connected layer to the network. Automatically flattens
        layer first if necessary

        Args:
            num_output_units: int. Number of units in output.
            activation_method: string. Specifies which of the available
                               activation methods in _activation() to use.
            use_batch_norm: bool. Whether batch normalization should be
                            applied to this layer. None to use CNNBuilder
                            object's default.
        """
        name = self._get_layer_name('dense')
        with tf.variable_scope(name):
            if len(self._top_layer.shape) != 2:
                new_shape = [self._top_layer.shape[0].value, -1]
                self._top_layer = tf.reshape(self._top_layer, new_shape)

            if use_batch_norm is None:
                use_batch_norm = self._use_batch_norm

            weight_decay_func = self._get_weight_decay_func()
            activation_func = (self._get_activation_func(activation_method) if
                               not use_batch_norm else None)
            use_bias = not use_batch_norm

            self._top_layer = tf.layers.dense(
                self._top_layer, num_output_units, activation_func, use_bias,
                kernel_regularizer=weight_decay_func,
                bias_regularizer=weight_decay_func)

            if use_batch_norm:
                self.batch_normalization()
                activation_func = self._get_activation_func(activation_method)
                self._top_layer = activation_func(self._top_layer)

            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self._top_layer)

    def dropout(self, keep_prob=0.5):
        """Adds dropout regularization layer; only applied during training.

        Args:
            keep_prob: float. Probability that any independent neuron will
                       be dropped, if currently in training phase.
        """
        name = self._get_layer_name('dropout')
        dropout = tf.layers.dropout(self._top_layer, keep_prob,
                                    training=self._is_training, name=name)
        self._top_layer = dropout

    def max_pooling(self, pool_size, strides=2, padding_mode=None):
        """Adds maximum pooling layer to network.

        Args:
            pool_size: int or (int, int). Height and width of 2D pooling
                       window. Single integer indicates height and width are
                       same size.
            strides: int or (int, int). Vertical and horizontal stride of
                     pooling window. Single integer indicates strides in both
                     dimensions are of same size.
            padding_mode: 'SAME' or 'VALID'. Padding mode to use. None to use
                          CNNBuilder object's default.
        """
        name = self._get_layer_name('maxpool')
        if padding_mode is None:
            padding_mode = self._padding_mode
        self._top_layer = tf.layers.max_pooling2d(
            self._top_layer, pool_size, strides, padding_mode,
            self._channel_pos, name)

    def _get_layer_name(self, prefix):
        """Creates unique name from prefix based on number of layers in CNN."""
        name = '{}_{}'.format(prefix, self._layer_counts[prefix])
        self._layer_counts[prefix] += 1
        return name

    def _get_weight_decay_func(self):
        if self._weight_decay_rate <= 0.0:
            return None

        def _weight_decay(tensor):
            return tf.multiply(tf.nn.l2_loss(tensor), self._weight_decay_rate,
                               name='weight_decay')

        return _weight_decay

    @staticmethod
    def _get_activation_func(method):
        method = method or 'linear'

        def _linear(tensor):
            return tf.identity(tensor, 'linear')

        activation_dict = {
            'linear': _linear,
            'relu': tf.nn.relu,
            'sigmoid': tf.nn.sigmoid,
            'softmax': tf.nn.softmax,
            'tanh': tf.nn.tanh
        }
        if method not in activation_dict:
            raise ValueError(
                "'{}' is not a valid activation method".format(method))
        return activation_dict[method]
