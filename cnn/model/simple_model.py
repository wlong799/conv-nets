# coding=utf-8
"""Simple model designed for classifying small images (e.g. MNIST or
CIFAR-10 datasets."""

import cnn


class SimpleModel(cnn.model.Model):
    """Simple CNN model."""

    def __init__(self, batch_size, num_classes):
        super(SimpleModel, self).__init__('simple', batch_size, num_classes)

    def inference(self, cnn_builder):
        """Simple CNN with convolution, pooling, and normalization."""
        cnn_builder.convolution(64, 3, 3)
        cnn_builder.convolution(64, 3, 3)
        cnn_builder.normalization()
        cnn_builder.max_pooling(3, 3)
        cnn_builder.convolution(128, 5, 5)
        cnn_builder.normalization()
        cnn_builder.max_pooling(3, 3)
        cnn_builder.reshape([self.batch_size, -1])
        cnn_builder.affine(512)
        cnn_builder.affine(256)
        logits, _ = cnn_builder.affine(self.num_classes)
        return logits
