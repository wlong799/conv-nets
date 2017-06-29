# coding=utf-8
"""Contains Model, a superclass for CNN model architectures"""

from .builder import CNNBuilder


class Model(object):
    """Superclass for all CNN model architectures.

    Classes that subclass model will override its inference method,
    to describe their own unique architectures. Subclasses should be placed
    in the cnn.model.implementations package.
    """

    def __init__(self, name, batch_size, num_classes):
        self.name = name
        self.batch_size = batch_size
        self.num_classes = num_classes

    # noinspection PyMethodMayBeStatic
    def inference(self, cnn_builder: CNNBuilder):
        """Builds network to perform inference.

        Args:
            cnn_builder: CNNBuilder class to use for building the network.

        Returns: logits: 2D Tensor [batch_size, num_classes] representing
                         the output from the last layer of the network.
                         These are the predictions for the class of each image
                         in the batch. No activation function (e.g. softmax)
                         should be applied to this layer, as the loss function
                         will perform this step itself.
        """
        raise ValueError("Inference must be implemented in derived classes.")
