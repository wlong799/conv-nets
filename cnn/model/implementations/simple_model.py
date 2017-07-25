# coding=utf-8
"""Simple CNN for classifying small images, such as the CIFAR-10 dataset."""

from cnn.model.model import Model
from cnn.model.builder import CNNBuilder


# noinspection PyMissingOrEmptyDocstring
class SimpleModel(Model):
    def __init__(self, batch_size, num_classes):
        super().__init__(batch_size, num_classes)

    @staticmethod
    def name():
        return 'simple'

    def inference(self, cnn_builder: CNNBuilder):
        """Simple CNN with convolution, pooling, and normalization."""
        cnn_builder.convolution(64, 3, 3)
        cnn_builder.convolution(64, 3, 3)
        cnn_builder.normalization()
        cnn_builder.max_pooling(3, 3)
        cnn_builder.convolution(128, 5, 5)
        cnn_builder.normalization()
        cnn_builder.max_pooling(3, 3)
        cnn_builder.reshape([self._batch_size, -1])
        cnn_builder.affine(512)
        cnn_builder.affine(256)
        logits, _ = cnn_builder.affine(self._num_classes,
                                       activation_method=None)
        return logits
