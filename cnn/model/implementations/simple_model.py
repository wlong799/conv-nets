# coding=utf-8
"""Simple CNN for classifying small images, such as the CIFAR-10 dataset."""

from cnn.model.model import Model
from cnn.model.builder import CNNBuilder


# noinspection PyMissingOrEmptyDocstring
class SimpleModel(Model):
    def __init__(self, batch_size, num_classes):
        super().__init__(batch_size, num_classes)

    @staticmethod
    def get_name():
        return 'simple'

    def _inference(self, cnn_builder: CNNBuilder):
        """Simple CNN with convolution, pooling, and fully connected layers."""
        if not (cnn_builder.padding_mode == 'SAME' and
                    cnn_builder.use_batch_norm):
            raise ValueError(
                "Model '{}' must be run with padding mode 'SAME' "
                "and batch normalization on.".format(self.get_name()))
        self._double_conv_layer(cnn_builder, 64)
        self._double_conv_layer(cnn_builder, 128)
        self._double_conv_layer(cnn_builder, 256)
        self._fully_connected_dropout_layer(cnn_builder, 512)
        self._fully_connected_dropout_layer(cnn_builder, 128)
        cnn_builder.dense(10, None)

    @staticmethod
    def _double_conv_layer(cnn_builder, num_filters):
        """Deeper networks with small filters have been shown to outperform
        shallower networks with large filters. 2 stacked 3x3 convolutions
        have the same FOV as a single 5x5 convolution. """
        cnn_builder.convolution(num_filters, 3)
        cnn_builder.convolution(num_filters, 3)
        cnn_builder.max_pooling(3)

    @staticmethod
    def _fully_connected_dropout_layer(cnn_builder, num_units):
        """Fully connected layer with dropout regularization to protect
        against over-fitting. Can use low rate of dropout because batch
        normalization provides a good deal of regularization on its own."""
        cnn_builder.dense(num_units)
        cnn_builder.dropout(0.1)
