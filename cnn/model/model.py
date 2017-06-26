# coding=utf-8
"""Contains Model, a superclass for CNN model architectures"""
import cnn


class Model(object):
    """Superclass for all CNN model architectures.

    Classes that subclass model will override its inference method,
    to describe their own unique architectures.
    """

    def __init__(self, name, batch_size, num_classes):
        self.name = name
        self.batch_size = batch_size
        self.num_classes = num_classes

    # noinspection PyMethodMayBeStatic
    def inference(self, cnn_builder):
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


def get_model(model_config: cnn.config.ModelConfig):
    """Returns model using the specified configuration. Should be edited as
    more model types are made available."""
    from cnn.model.simple_model import SimpleModel
    try:
        model = {
            'simple': SimpleModel(model_config.batch_size,
                                  model_config.num_classes)
        }[model_config.model_type]
    except KeyError as e:
        e.args = e.args or ('',)
        e.args += ("Model '{}' not available.".format(model_config.model_type))
        raise
    return model
