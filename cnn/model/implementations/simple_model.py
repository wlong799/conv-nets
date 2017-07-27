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
        cnn_builder.convolution(64, 3)
        cnn_builder.convolution(64, 3)
        cnn_builder.max_pooling(3)
        cnn_builder.convolution(128, 5)
        cnn_builder.max_pooling(3)
        cnn_builder.dense(512)
        cnn_builder.dense(256)
        cnn_builder.dense(10, None)
