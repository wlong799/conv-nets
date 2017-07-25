# coding=utf-8
"""Contains abstract superclass for CNN model architectures, as well as a
method to obtain the appropriate implementation class based on its name."""
import abc
from .builder import CNNBuilder


class Model(metaclass=abc.ABCMeta):
    """Abstract superclass for all CNN model architectures.

    Classes that subclass model will override its inference method,
    to describe their own unique architectures.
    """

    def __init__(self, batch_size, num_classes):
        self._batch_size = batch_size
        self._num_classes = num_classes

    @staticmethod
    @abc.abstractmethod
    def name():
        """Each Model subclass should have its own unique name. This name
        will be used to select the appropriate class according to the
        specified model configuration. """
        pass

    # noinspection PyMethodMayBeStatic
    @abc.abstractmethod
    def inference(self, cnn_builder: CNNBuilder):
        """Builds network to perform inference.

        Using CNNBuilder makes it easy to describe the model on a layer by
        layer basis. CNNBuilder will automatically connect the layers and
        perform all the background steps of data formatting, activation
        functions, etc. For instance, the following would be a valid model
        to specify in inference():

        cnn_builder.convolution(64, 3, 3)
        cnn_builder.max_pooling(3, 3)
        cnn_builder.normalization()
        cnn_builder.reshape([self.batch_size, -1])
        logits, _ = cnn_builder.affine(self.num_classes)

        Args:
            cnn_builder: CNNBuilder class to use for building the network.

        Returns: logits: 2D Tensor [batch_size, num_classes] representing
                         the output from the last layer of the network.
                         These are the predictions for the class of each image
                         in the batch. No activation function (e.g. softmax)
                         should be applied to this layer, as the loss function
                         will perform this step itself.
        """
        pass
