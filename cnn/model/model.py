# coding=utf-8
"""Abstract superclass for CNN model architectures."""
import abc
from .builder import CNNBuilder


class Model(metaclass=abc.ABCMeta):
    """Abstract superclass for all CNN model architectures."""

    def __init__(self, batch_size, num_classes):
        self._batch_size = batch_size
        self._num_classes = num_classes

    @staticmethod
    @abc.abstractmethod
    def get_name():
        """Each Model subclass should have its own unique name. This name
        will be used to select the appropriate class according to the
        specified model configuration. """
        pass

    def inference(self, cnn_builder: CNNBuilder):
        """Builds network to perform inference.

        Use CNNBuilder to easily build network layer by layer. Model
        architecture should be implemented in abstract _inference() method.

        Args:
            cnn_builder: CNNBuilder class to use for building the network.

        Returns: logits: 2D Tensor [batch_size, num_classes] representing
                         the output from the last layer of the network.
                         These are the predictions for the class of each image
                         in the batch. No activation function (e.g. softmax)
                         should be applied to this layer, as the loss function
                         will perform this step itself.
        """
        self._inference(cnn_builder)
        return cnn_builder.top_layer

    @abc.abstractmethod
    def _inference(self, cnn_builder):
        pass
